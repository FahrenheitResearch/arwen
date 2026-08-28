//! The parent-native boundary route: a lateral-boundary stream produced from
//! another run's own output, on its own mesh, on its own levels.
//!
//! ## What this route is for
//! A cascade — a coarse graded global driving a limited-area run driving a
//! small high-resolution swath — needs each level to hand the next one a
//! boundary.  Every part of that chain already existed except this: the
//! boundary producer consumed external-model first guesses and nothing else.
//!
//! ## The route, and what it costs
//! Two routes were open.  The first converts a parent's output into WPS
//! intermediates and reuses [`crate::lbc`]'s existing path unchanged.  The
//! second samples the parent onto the child's own cells and edges directly.
//! This module is the second.
//!
//! The first route's cost is not bookkeeping, it is fidelity, and it is
//! charged three times.  It resamples an unstructured mesh onto a regular
//! latitude/longitude array and then samples that array back onto an
//! unstructured mesh, so every field pays two interpolations where one would
//! do, and the intermediate grid's spacing — a number nobody has a principled
//! way to choose — sets how much of the parent survives.  It flattens the
//! parent's terrain-following column onto whatever level set the intermediate
//! carries, discarding the vertical structure the parent actually ran on.  And
//! it would then rebuild the boundary state through `init_atm_case_lbc`'s
//! first-guess pipeline: a hydrostatic rebalance that overwrites the parent's
//! own density with a re-derived one, and a vertical velocity reconstructed
//! from horizontal mass flux over sloping coordinate surfaces — which is the
//! right answer for a pressure-level first guess that has no vertical motion
//! in it, and the wrong answer for a parent that does.  Self-nesting exists to
//! carry the parent's convection to the child's boundary; a route that
//! discards `w` and re-derives `rho` gives that away at the door.
//!
//! What the second route costs is one new geometry operator — unstructured
//! source onto unstructured target, in [`crate::lbc::sphere`] — and the
//! vertical remap below.  That is the whole bill, and it buys a transfer that
//! is exactly the identity when the child mesh is the parent mesh.
//!
//! ## Why no state is rebuilt
//! Measured, not assumed.  Against the 2026-08-25 regional oracle at x1, the
//! native case-7 initial state and the native case-9 boundary file at the same
//! valid time agree to `0.0` on `theta` and `qv` — bit for bit — and to
//! `4.9e-7` relative on `rho`.  The lbc stream's contents *are* the model's own
//! decoupled state: `lbc_theta` is `theta`, `lbc_rho` is `rho`, `lbc_qv` is
//! `qv`, `lbc_u` is `u`, `lbc_w` is `w`.  So the parent-native route carries
//! them across and remaps them; it does not rebuild them.  The only place it
//! departs from the first-guess pipeline's arithmetic is where that pipeline
//! was reconstructing what a parent already has.
//!
//! ## The two divergences from the first-guess route, both deliberate
//! * **Density is remapped in the log.**  `rho` falls off exponentially with
//!   height; interpolating it linearly in `z` biases every level low between
//!   the parent's layers.  `init_atm_case_lbc` treats pressure exactly this
//!   way for exactly this reason.
//! * **A one-cell edge takes its height column from the cell that exists.**
//!   The first-guess route reproduces native's arithmetic at the outer edges
//!   of a culled mesh, where a stored-zero `cellsOnEdge` entry contributes a
//!   column of zeros to the four-point mean that sets the target height.  That
//!   is native's own behaviour and the pin requires it there.  Here it would
//!   halve the target height at exactly the outermost boundary ring and sample
//!   the parent's column at the wrong altitude, so the mean is taken over the
//!   cells that exist.  The count is in the receipt.

#![allow(clippy::needless_range_loop)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::error::{MpasError, MpasResult};
use crate::init::capsule::{self, MeshGeometry, VerticalMetrics};
use crate::init::hinterp::Underflow;
use crate::init::vinterp::{vertical_interp, Extrap};
use crate::lbc::source::{Registry, Role, SourceRow, StateKind};
use crate::lbc::sphere::{CellOperator, EdgeOperator, Miss, SphereSampler};
use crate::lbc::{emit, seconds_between, BoundaryInterval, LbcConfig};

/// `config_theta_adv_order` and `config_coef_3rd_order` as the v8.4.1 registry
/// defaults them.  This route advects nothing, so the header carries the
/// declared defaults and `gpuwm_provenance` records that the route was
/// parent-native — the numbers in the file came from a parent's state, not
/// from a first-guess pipeline with these switches in it.
const REGISTRY_THETA_ADV_ORDER: i32 = 3;
const REGISTRY_COEF_3RD_ORDER: f32 = 0.25;

/// Everything the caller must state for a parent-native run.
#[derive(Debug, Clone)]
pub struct ParentConfig {
    /// The child's initial-conditions file: the target mesh, its layer
    /// heights and its lineage all come from here.
    pub grid_path: PathBuf,
    /// The parent's mesh and layer heights, when its state frames do not
    /// carry them.  A forecast history stream normally does not.
    pub parent_grid: Option<PathBuf>,
    pub out_dir: PathBuf,
    pub start_time: String,
    pub stop_time: String,
    /// One boundary time each, naming the parent frame that carries it.
    pub intervals: Vec<BoundaryInterval>,
    pub fg_interval_seconds: i64,
    /// The driving-source row: what the parent's files are called and how
    /// they are spelled.
    pub source_row: String,
    pub registry_path: Option<PathBuf>,
    pub attr_overrides: BTreeMap<String, serde_json::Value>,
    pub provenance: String,
    /// Turn the coincidence snap off, so a degenerate transfer's accuracy can
    /// be measured rather than asserted.
    pub without_snap: bool,
}

/// Receipt for one produced boundary time.
#[derive(Debug, Default, serde::Serialize)]
pub struct ParentIntervalReceipt {
    pub valid_time: String,
    pub parent_path: String,
    pub parent_sha256: String,
    pub out_path: String,
    pub time_seconds: f64,
    /// The parent frame's own record of its valid time, when the row maps one.
    pub parent_valid_time: Option<String>,
    pub valid_time_verified: bool,
    /// Roles the row maps that the parent frame did not carry, written as
    /// zeros.  Only the optional condensate slots may appear here.
    pub roles_absent: Vec<String>,
    /// How far the deepest child column reached below the parent's lowest
    /// layer midpoint, in metres.  Sub-terrain extrapolation is what every
    /// nest does over resolved valleys; this says how much of it happened.
    pub max_metres_below_parent_column: f64,
    /// Values that arrived below zero in a mixing ratio and were set to zero.
    /// The parent's own transport leaves a scattering of them; carrying
    /// negative water to a child's boundary would drive it towards a state
    /// that cannot exist.
    pub negative_mixing_ratios_clamped: usize,
    pub out_sha256: String,
    pub seconds: f64,
}

/// The whole run's receipt.
#[derive(Debug, Default, serde::Serialize)]
pub struct ParentReceipt {
    pub route: String,
    pub source_row: String,
    pub source_row_notes: String,
    pub grid_path: String,
    pub parent_grid_path: Option<String>,
    pub child_cells: usize,
    pub child_edges: usize,
    pub child_levels: usize,
    pub parent_cells: usize,
    pub parent_edges: usize,
    pub parent_levels: usize,
    /// Child cells whose operator collapsed onto a single parent cell, and
    /// child edges that took a parent edge's own component.
    pub cells_snapped: usize,
    pub edges_snapped: usize,
    /// Child cells placed in the parent's outermost cells rather than in its
    /// dual triangulation: first-order there where the interior is
    /// second-order.
    pub cells_in_skirt: usize,
    /// Child edges whose height column came from one adjacent cell because the
    /// other is the stored-zero garbage cell.
    pub one_cell_edges: usize,
    /// The worst conditioning of any child edge's local wind fit.
    pub worst_edge_condition: f64,
    pub coincidence_snap: bool,
    pub intervals: Vec<ParentIntervalReceipt>,
    pub seconds: f64,
}

/// The parent's mesh, layer heights and the operators that map it onto the
/// child.  Built once and reused for every boundary time, which is what makes
/// a long series cheap.
struct Transfer {
    parent_cells: usize,
    parent_edges: usize,
    parent_levels: usize,
    /// One operator per child cell.
    cell_ops: Vec<CellOperator>,
    /// One operator per child edge, for the edge-normal wind.
    edge_ops: Vec<EdgeOperator>,
    /// Source layer-midpoint heights at each child cell, `[cell][k]`.
    cell_src_mid: Vec<Vec<f32>>,
    /// Source layer-interface heights at each child cell, `[cell][k]`.
    cell_src_iface: Vec<Vec<f32>>,
    /// Source layer-midpoint heights at each child edge, `[edge][k]`.
    edge_src_mid: Vec<Vec<f32>>,
    /// Target layer-midpoint heights at each child edge, `[edge][k]`.
    edge_tgt_mid: Vec<Vec<f32>>,
    cells_snapped: usize,
    edges_snapped: usize,
    cells_in_skirt: usize,
    worst_edge_condition: f64,
    one_cell_edges: usize,
    max_metres_below: f64,
}

/// A file pair the row's roles resolve against: the state frame first, the
/// companion grid second.
struct RoleReader {
    state: netcrust::File,
    state_path: PathBuf,
    grid: Option<netcrust::File>,
    grid_path: Option<PathBuf>,
}

impl RoleReader {
    fn open(state_path: &Path, grid_path: Option<&Path>) -> MpasResult<RoleReader> {
        if !state_path.exists() {
            return Err(MpasError::Refusal(format!(
                "the parent frame {} does not exist.  Either the parent run never reached this \
                 boundary time, or its history was written somewhere else; producing this time \
                 from a neighbouring frame would put the wrong hour on the child's boundary",
                state_path.display()
            )));
        }
        let state = netcrust::File::open(state_path)?;
        let grid = match grid_path {
            None => None,
            Some(p) => {
                if !p.exists() {
                    return Err(MpasError::Refusal(format!(
                        "the parent grid {} does not exist; the parent's mesh and layer heights \
                         have nowhere to come from",
                        p.display()
                    )));
                }
                Some(netcrust::File::open(p)?)
            }
        };
        Ok(RoleReader {
            state,
            state_path: state_path.to_path_buf(),
            grid,
            grid_path: grid_path.map(Path::to_path_buf),
        })
    }

    fn where_from(&self) -> String {
        match &self.grid_path {
            None => self.state_path.display().to_string(),
            Some(g) => format!("{} or {}", self.state_path.display(), g.display()),
        }
    }

    /// Read a role's variable as f64, first record if it has a record axis.
    fn read(&self, row: &SourceRow, role: Role) -> MpasResult<Vec<f64>> {
        let name = row.require(role)?;
        self.read_named(row, role, name)
    }

    fn read_named(&self, row: &SourceRow, role: Role, name: &str) -> MpasResult<Vec<f64>> {
        if let Ok(a) = self.state.read_array_f64_first_record_or_all(name) {
            return Ok(a.into_values());
        }
        if let Some(grid) = &self.grid {
            if let Ok(a) = grid.read_array_f64_first_record_or_all(name) {
                return Ok(a.into_values());
            }
        }
        Err(MpasError::Refusal(format!(
            "the driving-source row \"{}\" maps the role {} to the variable \"{name}\", and no \
             such variable is readable in {}: {}",
            row.name,
            role.label(),
            self.where_from(),
            role.because()
        )))
    }

    /// Read an optional role: `None` when the row maps it but the file does
    /// not carry it.  Refuses only when the row maps nothing at all is not an
    /// error here — an unmapped optional role is simply absent.
    fn read_optional(&self, row: &SourceRow, role: Role) -> Option<Vec<f64>> {
        let name = row.name_of(role)?;
        if let Ok(a) = self.state.read_array_f64_first_record_or_all(name) {
            return Some(a.into_values());
        }
        if let Some(grid) = &self.grid {
            if let Ok(a) = grid.read_array_f64_first_record_or_all(name) {
                return Some(a.into_values());
            }
        }
        None
    }

    fn dimension(&self, name: &str) -> Option<usize> {
        self.state
            .dimension(name)
            .map(|d| d.len())
            .or_else(|| self.grid.as_ref().and_then(|g| g.dimension(name)).map(|d| d.len()))
    }

    /// The parent's own valid time, when the row maps one and the frame
    /// carries it.  A stream stamps its time as text, so this goes through the
    /// character reader rather than the promoting numeric one.
    fn valid_time(&self, row: &SourceRow) -> Option<String> {
        let name = row.name_of(Role::ValidTime)?;
        let mut text = crate::lbc::compare::read_char_variable(&self.state_path, name)
            .ok()
            .flatten();
        if text.is_none() {
            if let Some(grid) = &self.grid_path {
                text = crate::lbc::compare::read_char_variable(grid, name)
                    .ok()
                    .flatten();
            }
        }
        let text = text?;
        let trimmed = text.trim_end_matches(['\0', ' ']).trim();
        if trimmed.len() < 19 {
            return None;
        }
        Some(trimmed[..19].to_string())
    }
}

fn split_columns(flat: Vec<f64>, levels: usize, n: usize, what: &str) -> MpasResult<Vec<Vec<f32>>> {
    if flat.len() != levels * n {
        return Err(MpasError::Refusal(format!(
            "{what} holds {} value(s); {n} column(s) of {levels} level(s) is {}.  A count that \
             does not divide is the point-major/level-major transpose, which reads as a plausible \
             file full of the wrong numbers",
            flat.len(),
            levels * n
        )));
    }
    Ok((0..n)
        .map(|i| {
            flat[i * levels..(i + 1) * levels]
                .iter()
                .map(|&v| v as f32)
                .collect()
        })
        .collect())
}

fn split_connectivity(flat: &[f64], width: usize, n: usize) -> Vec<Vec<usize>> {
    (0..n)
        .map(|i| {
            (0..width)
                .map(|j| {
                    let v = flat[i * width + j].round();
                    if v > 0.0 {
                        v as usize
                    } else {
                        0
                    }
                })
                .collect()
        })
        .collect()
}

/// Build the whole boundary series from a parent's own output.
pub fn build_from_parent(cfg: &ParentConfig) -> MpasResult<ParentReceipt> {
    let started = std::time::Instant::now();

    if cfg.intervals.is_empty() {
        return Err(MpasError::Refusal(
            "no boundary intervals were given.  Each --interval names one valid time and the \
             parent frame carrying it; without at least one there is nothing to produce"
                .to_string(),
        ));
    }

    let registry = match &cfg.registry_path {
        None => Registry::built_in()?,
        Some(p) => Registry::with_file(p)?,
    };
    let row = registry.row(&cfg.source_row)?.clone();
    if row.state != StateKind::Prognostic {
        return Err(MpasError::Refusal(format!(
            "the driving-source row \"{}\" carries a {:?} state, and the parent-native route \
             transfers a prognostic one.  A first-guess source goes through the first-guess \
             route, which rebuilds the boundary state instead of carrying it",
            row.name, row.state
        )));
    }

    let child_mesh = capsule::read_mesh_geometry(&cfg.grid_path)?;
    let child_metrics =
        capsule::read_vertical_metrics(&cfg.grid_path, child_mesh.n_cells, child_mesh.n_edges)?;
    let header = emit::HeaderSource::from_grid(&cfg.grid_path)?;

    let first = &cfg.intervals[0];
    let transfer = build_transfer(cfg, &row, &first.met_path, &child_mesh, &child_metrics)?;

    let mut receipt = ParentReceipt {
        route: "parent-native".to_string(),
        source_row: row.name.clone(),
        source_row_notes: row.notes.clone(),
        grid_path: cfg.grid_path.display().to_string(),
        parent_grid_path: cfg.parent_grid.as_ref().map(|p| p.display().to_string()),
        child_cells: child_mesh.n_cells,
        child_edges: child_mesh.n_edges,
        child_levels: child_metrics.n_vert_levels,
        parent_cells: transfer.parent_cells,
        parent_edges: transfer.parent_edges,
        parent_levels: transfer.parent_levels,
        cells_snapped: transfer.cells_snapped,
        edges_snapped: transfer.edges_snapped,
        cells_in_skirt: transfer.cells_in_skirt,
        one_cell_edges: transfer.one_cell_edges,
        worst_edge_condition: transfer.worst_edge_condition,
        coincidence_snap: !cfg.without_snap,
        ..Default::default()
    };

    std::fs::create_dir_all(&cfg.out_dir)?;

    let lbc_cfg = LbcConfig {
        grid_path: cfg.grid_path.clone(),
        out_dir: cfg.out_dir.clone(),
        start_time: cfg.start_time.clone(),
        stop_time: cfg.stop_time.clone(),
        intervals: cfg.intervals.clone(),
        n_fg_levels: transfer.parent_levels,
        extrap_airtemp: Extrap::Constant,
        use_spechumd: false,
        theta_adv_order: REGISTRY_THETA_ADV_ORDER,
        coef_3rd_order: REGISTRY_COEF_3RD_ORDER,
        fg_interval_seconds: cfg.fg_interval_seconds,
        oned_underflow: Underflow::Preserve,
        attr_overrides: cfg.attr_overrides.clone(),
        provenance: format!(
            "{} | route=parent-native source-row={} parent-grid={} one-cell-edges={}",
            cfg.provenance,
            row.name,
            cfg.parent_grid
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "in the frame".to_string()),
            transfer.one_cell_edges
        ),
    };

    for interval in &cfg.intervals {
        let one = build_one_time(
            cfg,
            &lbc_cfg,
            &row,
            &transfer,
            &child_mesh,
            &child_metrics,
            &header,
            interval,
        )?;
        receipt.intervals.push(one);
    }

    receipt.seconds = started.elapsed().as_secs_f64();
    Ok(receipt)
}

/// Read the parent's mesh and layer heights and build every operator.
fn build_transfer(
    cfg: &ParentConfig,
    row: &SourceRow,
    first_frame: &Path,
    child: &MeshGeometry,
    child_metrics: &VerticalMetrics,
) -> MpasResult<Transfer> {
    let reader = RoleReader::open(first_frame, cfg.parent_grid.as_deref())?;

    let parent_cells = reader.dimension("nCells").ok_or_else(|| {
        MpasError::Refusal(format!(
            "{} declares no nCells dimension; the parent-native route needs a source mesh, and a \
             file without one is not a mesh-borne state",
            reader.where_from()
        ))
    })?;
    let parent_edges = reader.dimension("nEdges").ok_or_else(|| {
        MpasError::Refusal(format!(
            "{} declares no nEdges dimension; the parent's edge-normal wind has nowhere to live",
            reader.where_from()
        ))
    })?;
    let parent_levels = reader.dimension("nVertLevels").ok_or_else(|| {
        MpasError::Refusal(format!(
            "{} declares no nVertLevels dimension",
            reader.where_from()
        ))
    })?;
    if parent_levels < 2 {
        return Err(MpasError::Refusal(format!(
            "the parent carries {parent_levels} vertical level(s); a column cannot be remapped \
             from fewer than two"
        )));
    }
    let vertex_degree = reader.dimension("vertexDegree").unwrap_or(3);
    let max_edges = reader.dimension("maxEdges").ok_or_else(|| {
        MpasError::Refusal(format!(
            "{} declares no maxEdges dimension; the parent's edge neighbourhood cannot be read \
             at a stride nobody stated",
            reader.where_from()
        ))
    })?;

    let cell_lat: Vec<f64> = reader.read(row, Role::CellLatitude)?;
    let cell_lon: Vec<f64> = reader.read(row, Role::CellLongitude)?;
    let edge_lat: Vec<f64> = reader.read(row, Role::EdgeLatitude)?;
    let edge_lon: Vec<f64> = reader.read(row, Role::EdgeLongitude)?;
    let edge_angle: Vec<f64> = reader.read(row, Role::EdgeNormalAngle)?;
    if cell_lat.len() != parent_cells || edge_lat.len() != parent_edges {
        return Err(MpasError::Refusal(format!(
            "the parent's coordinates cover {} cell(s) and {} edge(s); its dimensions declare \
             {parent_cells} and {parent_edges}.  The mesh and the state disagree about their own \
             size, and every weight built from one would address the other's memory",
            cell_lat.len(),
            edge_lat.len()
        )));
    }

    let cov_raw = reader.read(row, Role::CellsOnVertex)?;
    let cov: Vec<i64> = cov_raw.iter().map(|v| v.round() as i64).collect();
    let eoc_raw = reader.read(row, Role::EdgesOnCell)?;
    let coc_raw = reader.read(row, Role::CellsOnCell)?;
    let neoc_raw = reader.read(row, Role::EdgesPerCell)?;
    let edges_on_cell = split_connectivity(&eoc_raw, max_edges, parent_cells);
    let cells_on_cell = split_connectivity(&coc_raw, max_edges, parent_cells);
    let n_edges_on_cell: Vec<usize> = neoc_raw
        .iter()
        .map(|v| (v.round().max(0.0) as usize).min(max_edges))
        .collect();

    let zgrid_flat = reader.read(row, Role::InterfaceHeight)?;
    let parent_zgrid = split_columns(
        zgrid_flat,
        parent_levels + 1,
        parent_cells,
        row.require(Role::InterfaceHeight)?,
    )?;
    let max_z = parent_zgrid
        .iter()
        .flat_map(|c| c.iter())
        .fold(f32::MIN, |a, &b| a.max(b));
    if max_z <= 0.0 {
        return Err(MpasError::Refusal(format!(
            "every layer height in the parent's {} is zero or below; the parent's vertical \
             coordinate is missing, and every child column would be remapped against height zero",
            row.require(Role::InterfaceHeight)?
        )));
    }
    // The remap walks each source column assuming it ascends.  A column that
    // does not is not a rounding problem: every level would be looked up in
    // the wrong bracket and the answer would still be finite.
    for (c, col) in parent_zgrid.iter().enumerate() {
        for k in 0..parent_levels {
            if !(col[k + 1] > col[k]) {
                return Err(MpasError::Refusal(format!(
                    "the parent's layer heights do not ascend at cell {} between levels {k} and \
                     {} ({} m then {} m).  Every target height would be bracketed against a \
                     column that folds back on itself, and the result would be finite, plausible \
                     and wrong",
                    c + 1,
                    k + 1,
                    col[k],
                    col[k + 1]
                )));
            }
        }
    }

    let mut sampler = SphereSampler::build(
        &cell_lat,
        &cell_lon,
        &cov,
        vertex_degree,
        &edge_lat,
        &edge_lon,
        &edge_angle,
        &edges_on_cell,
        &n_edges_on_cell,
        &cells_on_cell,
    )?;
    if cfg.without_snap {
        sampler = sampler.without_snap();
    }

    // Parent layer midpoints, one column per parent cell.
    let parent_mid: Vec<Vec<f32>> = parent_zgrid
        .iter()
        .map(|col| {
            (0..parent_levels)
                .map(|k| 0.5 * (col[k] + col[k + 1]))
                .collect()
        })
        .collect();

    // Child cell operators.
    let n_cells = child.n_cells;
    let cell_ops: Vec<MpasResult<CellOperator>> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            let lat = child.lat_cell[c] as f64;
            let lon = child.lon_cell[c] as f64;
            sampler
                .cell_weights(lat, lon)
                .map_err(|miss| MpasError::Refusal(outside_message("cell", c, lat, lon, miss)))
        })
        .collect();
    let cell_ops = collect_containment(cell_ops, "cell", n_cells)?;

    // Child edge operators: the wind fit only.  Containment is a statement
    // about the child's CELLS — an edge between two contained cells is itself
    // contained — and the edge's height columns come from those same cells
    // below, so nothing here needs to place the edge point in a triangle.
    let n_edges = child.n_edges;
    let edge_ops: Vec<MpasResult<EdgeOperator>> = (0..n_edges)
        .into_par_iter()
        .map(|e| {
            let lat = child.lat_edge[e] as f64;
            let lon = child.lon_edge[e] as f64;
            sampler
                .edge_operator(lat, lon, child.angle_edge[e] as f64)
                .ok_or_else(|| {
                    MpasError::Refusal(format!(
                        "no well-posed wind fit exists at child edge {} ({:.4}N {:.4}E): the \
                         parent edges around it do not span two directions, so the earth-relative \
                         wind cannot be recovered from their normal components and any answer \
                         here would be one direction of the flow asserted as both",
                        e + 1,
                        lat * crate::init::DEG_PER_RAD as f64,
                        lon * crate::init::DEG_PER_RAD as f64
                    ))
                })
        })
        .collect();
    let edge_ops = collect_containment(edge_ops, "edge", n_edges)?;

    // Source and target height columns, fixed for the whole series.
    let nz = child_metrics.n_vert_levels;
    let cell_src_mid: Vec<Vec<f32>> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            (0..parent_levels)
                .map(|k| cell_ops[c].apply_column(&parent_mid, k))
                .collect()
        })
        .collect();
    let cell_src_iface: Vec<Vec<f32>> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            (0..parent_levels + 1)
                .map(|k| cell_ops[c].apply_column(&parent_zgrid, k))
                .collect()
        })
        .collect();

    // An edge's source and target height columns are BOTH the mean over the
    // child cells that edge actually has, of those cells' own columns — the
    // source one of the parent heights sampled there, the target one of the
    // child's own heights.  Building them by the same rule from the same cells
    // is what makes the vertical remap a pure shift: sample the edge point
    // separately and the two columns disagree by the difference between a
    // three-cell barycentric mean and a two-cell one, which on a degenerate
    // nest is a shift where there should be none.
    //
    // Where one side is the stored-zero garbage cell the mean is over the cell
    // that exists, on both sides of the remap.  See this module's header for
    // why that departs from the first-guess route's four-point mean.
    let mut one_cell_edges = 0usize;
    let edge_sides: Vec<Vec<usize>> = (0..n_edges)
        .map(|e| {
            let [c1, c2] = child.cells_on_edge[e];
            let ok = |c: usize| c >= 1 && c <= n_cells;
            let mut sides: Vec<usize> = Vec::with_capacity(2);
            if ok(c1) {
                sides.push(c1 - 1);
            }
            if ok(c2) {
                sides.push(c2 - 1);
            }
            if sides.len() < 2 {
                one_cell_edges += 1;
            }
            sides
        })
        .collect();
    let column_mean = |sides: &[usize], levels: usize, of: &dyn Fn(usize, usize) -> f32| {
        let inv = 1.0 / sides.len().max(1) as f32;
        (0..levels)
            .map(|k| {
                let mut acc = 0.0f32;
                for &c in sides {
                    acc += of(c, k);
                }
                acc * inv
            })
            .collect::<Vec<f32>>()
    };
    let edge_src_mid: Vec<Vec<f32>> = (0..n_edges)
        .map(|e| {
            column_mean(&edge_sides[e], parent_levels, &|c, k| cell_src_mid[c][k])
        })
        .collect();
    let edge_tgt_mid: Vec<Vec<f32>> = (0..n_edges)
        .map(|e| {
            column_mean(&edge_sides[e], nz, &|c, k| {
                0.5 * (child_metrics.zgrid[c][k] + child_metrics.zgrid[c][k + 1])
            })
        })
        .collect();
    if one_cell_edges == n_edges && n_edges > 0 {
        return Err(MpasError::Refusal(
            "every child edge names a stored-zero cell on both sides; the child's cellsOnEdge \
             carries no valid cell anywhere, so no target height column can be formed"
                .to_string(),
        ));
    }

    // The vertical compatibility check, and the sub-terrain exposure.
    let mut above = 0usize;
    let mut worst_above = 0.0f64;
    let mut worst_cell = 0usize;
    let mut max_below = 0.0f64;
    for c in 0..n_cells {
        let src_top = cell_src_iface[c][parent_levels] as f64;
        let tgt_top = child_metrics.zgrid[c][nz] as f64;
        if tgt_top > src_top {
            above += 1;
            if tgt_top - src_top > worst_above {
                worst_above = tgt_top - src_top;
                worst_cell = c;
            }
        }
        let below = cell_src_mid[c][0] as f64
            - 0.5 * (child_metrics.zgrid[c][0] + child_metrics.zgrid[c][1]) as f64;
        if below > max_below {
            max_below = below;
        }
    }
    if above > 0 {
        return Err(MpasError::Refusal(format!(
            "{above} child column(s) reach above the parent's model top; the worst is child cell \
             {} by {worst_above:.1} m.  Above the parent's top there is no parent atmosphere, so \
             those levels would be filled by extrapolating off the end of the parent's column — \
             a boundary invented rather than driven.  Give the child a model top at or below the \
             parent's, or drive it from a parent that reaches higher",
            worst_cell + 1
        )));
    }

    let cells_snapped = cell_ops.iter().filter(|o| o.snapped).count();
    let edges_snapped = edge_ops.iter().filter(|o| o.snapped).count();
    let cells_in_skirt = cell_ops.iter().filter(|o| o.skirt).count();
    let worst_edge_condition = edge_ops
        .iter()
        .filter(|o| !o.snapped)
        .map(|o| o.condition)
        .fold(0.0f64, f64::max);

    Ok(Transfer {
        parent_cells,
        parent_edges,
        parent_levels,
        cell_ops,
        edge_ops,
        cell_src_mid,
        cell_src_iface,
        edge_src_mid,
        edge_tgt_mid,
        cells_snapped,
        edges_snapped,
        cells_in_skirt,
        worst_edge_condition,
        one_cell_edges,
        max_metres_below: max_below,
    })
}

/// Mean Earth radius, for turning a chord on the unit sphere into metres a
/// person can read.  The refusal only ever quotes it, so the exact sphere the
/// mesh was built on would move the number by a tenth of a percent.
const EARTH_RADIUS_M: f64 = 6_371_229.0;

/// What "outside the parent" means, with the distance that made it so.
fn outside_message(what: &str, index: usize, lat: f64, lon: f64, miss: Miss) -> String {
    format!(
        "child {what} {} at {:.4}N {:.4}E lies outside the parent's domain: it is {:.0} m from \
         the nearest parent cell, which reaches {:.0} m.  A child must be contained in the \
         parent that drives it; filling this point would extrapolate a boundary the parent never \
         computed, and the child would relax its outermost ring towards a state nothing produced",
        index + 1,
        lat * crate::init::DEG_PER_RAD as f64,
        lon * crate::init::DEG_PER_RAD as f64,
        miss.distance * EARTH_RADIUS_M,
        miss.reach * EARTH_RADIUS_M
    )
}

/// Collapse a parallel result vector, reporting how many targets missed rather
/// than only the first.
fn collect_containment<T>(
    items: Vec<MpasResult<T>>,
    what: &str,
    total: usize,
) -> MpasResult<Vec<T>> {
    let misses = items.iter().filter(|r| r.is_err()).count();
    if misses > 0 {
        let first = items
            .into_iter()
            .find_map(|r| r.err())
            .expect("a miss was counted");
        return Err(MpasError::Refusal(format!(
            "{misses} of {total} child {what}(s) could not be placed in the parent.  {first}"
        )));
    }
    Ok(items.into_iter().map(|r| r.expect("no misses")).collect())
}

#[allow(clippy::too_many_arguments)]
fn build_one_time(
    cfg: &ParentConfig,
    lbc_cfg: &LbcConfig,
    row: &SourceRow,
    transfer: &Transfer,
    child: &MeshGeometry,
    child_metrics: &VerticalMetrics,
    header: &emit::HeaderSource,
    interval: &BoundaryInterval,
) -> MpasResult<ParentIntervalReceipt> {
    let started = std::time::Instant::now();
    let n_cells = child.n_cells;
    let n_edges = child.n_edges;
    let nz = child_metrics.n_vert_levels;
    let nzp1 = nz + 1;
    let pz = transfer.parent_levels;

    let reader = RoleReader::open(&interval.met_path, cfg.parent_grid.as_deref())?;

    // The parent frame must be the mesh the operators were built for.
    for (name, want) in [
        ("nCells", transfer.parent_cells),
        ("nEdges", transfer.parent_edges),
        ("nVertLevels", transfer.parent_levels),
    ] {
        let got = reader.dimension(name).unwrap_or(0);
        if got != want {
            return Err(MpasError::Refusal(format!(
                "the parent frame {} declares {name} = {got}; the first frame in this series \
                 declared {want}.  A parent that changes mesh mid-series would have every \
                 boundary time after the change sampled through the wrong weights",
                interval.met_path.display()
            )));
        }
    }

    let mut receipt = ParentIntervalReceipt {
        valid_time: interval.valid_time.clone(),
        parent_path: interval.met_path.display().to_string(),
        parent_sha256: crate::sha256_file(&interval.met_path)?,
        max_metres_below_parent_column: transfer.max_metres_below,
        ..Default::default()
    };

    // The parent frame's own record of when it is, when the row maps one.
    match reader.valid_time(row) {
        Some(stamp) => {
            let normalised = stamp.replace('.', ":");
            receipt.parent_valid_time = Some(stamp.clone());
            if normalised != interval.valid_time {
                return Err(MpasError::Refusal(format!(
                    "the parent frame {} says it is valid at {stamp}, and it was named as the \
                     boundary time {}.  Writing it anyway would stamp the child's boundary with \
                     an hour the parent never held there",
                    interval.met_path.display(),
                    interval.valid_time
                )));
            }
            receipt.valid_time_verified = true;
        }
        None => {
            if row.name_of(Role::ValidTime).is_some() {
                return Err(MpasError::Refusal(format!(
                    "the driving-source row \"{}\" says this source records its own valid time in \
                     \"{}\", and {} carries no readable stamp there.  A row that claims a \
                     verifiable time and then cannot be verified is worse than one that never \
                     claimed it",
                    row.name,
                    row.name_of(Role::ValidTime).unwrap_or("?"),
                    interval.met_path.display()
                )));
            }
            receipt.valid_time_verified = false;
        }
    }

    // The parent's state, in its own spelling.
    let read_cells = |role: Role, levels: usize| -> MpasResult<Vec<Vec<f32>>> {
        let flat = reader.read(row, role)?;
        split_columns(flat, levels, transfer.parent_cells, row.require(role)?)
    };
    let theta_p = read_cells(Role::DryPotentialTemperature, pz)?;
    let rho_p = read_cells(Role::DryDensity, pz)?;
    let qv_p = read_cells(Role::VapourMixingRatio, pz)?;
    let w_p = read_cells(Role::VerticalVelocity, pz + 1)?;

    let optional_cells = |role: Role| -> MpasResult<Option<Vec<Vec<f32>>>> {
        match reader.read_optional(row, role) {
            None => Ok(None),
            Some(flat) => Ok(Some(split_columns(
                flat,
                pz,
                transfer.parent_cells,
                row.require(role)?,
            )?)),
        }
    };
    let qc_p = optional_cells(Role::CloudMixingRatio)?;
    let qr_p = optional_cells(Role::RainMixingRatio)?;
    for (role, present) in [
        (Role::CloudMixingRatio, qc_p.is_some()),
        (Role::RainMixingRatio, qr_p.is_some()),
    ] {
        if !present {
            receipt.roles_absent.push(role.label().to_string());
        }
    }

    let u_flat = reader.read(row, Role::EdgeNormalWind)?;
    let u_p = split_columns(
        u_flat,
        pz,
        transfer.parent_edges,
        row.require(Role::EdgeNormalWind)?,
    )?;

    // Log density: remapped in the log because density is exponential in
    // height, then exponentiated back.
    let mut log_rho_p = rho_p;
    let mut nonpositive = 0usize;
    for col in log_rho_p.iter_mut() {
        for v in col.iter_mut() {
            if *v > 0.0 {
                *v = v.ln();
            } else {
                nonpositive += 1;
                *v = f32::MIN_POSITIVE.ln();
            }
        }
    }
    if nonpositive > 0 {
        return Err(MpasError::Refusal(format!(
            "the parent's dry density is zero or negative at {nonpositive} point(s).  Density is \
             remapped in the log because it falls off exponentially with height; a non-positive \
             value there is not a rounding problem, it is a parent state that has already gone \
             wrong, and carrying it to a child's boundary would spread it"
        )));
    }

    // The vertical remap, per child cell.
    type Cols = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, usize);
    let cells: Vec<Cols> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            let op = &transfer.cell_ops[c];
            let src = &transfer.cell_src_mid[c];
            let src_iface = &transfer.cell_src_iface[c];
            let mut scratch = vec![0.0f32; pz];
            let target = |k: usize| {
                0.5 * (child_metrics.zgrid[c][k] + child_metrics.zgrid[c][k + 1])
            };

            let remap = |source: &[Vec<f32>], extrap: Extrap, scratch: &mut Vec<f32>| {
                for k in 0..pz {
                    scratch[k] = op.apply_column(source, k);
                }
                (0..nz)
                    .map(|k| {
                        vertical_interp(target(k), src, scratch, extrap)
                            .unwrap_or_else(|_| scratch[pz - 1])
                    })
                    .collect::<Vec<f32>>()
            };

            // A mixing ratio that arrives negative — the parent's own
            // transport leaves a scattering of values a few parts in 1e8 below
            // zero — is set to zero rather than carried.  Negative water is
            // not a state the child can be driven towards, and every count is
            // reported so the clamp is visible rather than quiet.
            let mut clamped = 0usize;
            let clamp = |v: f32, clamped: &mut usize| {
                if v < 0.0 {
                    *clamped += 1;
                    0.0
                } else {
                    v
                }
            };

            let theta = remap(&theta_p, Extrap::Linear, &mut scratch);
            let log_rho = remap(&log_rho_p, Extrap::Linear, &mut scratch);
            let rho: Vec<f32> = log_rho.iter().map(|v| v.exp()).collect();
            let qv: Vec<f32> = remap(&qv_p, Extrap::Constant, &mut scratch)
                .into_iter()
                .map(|v| clamp(v, &mut clamped))
                .collect();
            let qc: Vec<f32> = match &qc_p {
                None => vec![0.0; nz],
                Some(src) => remap(src, Extrap::Constant, &mut scratch)
                    .into_iter()
                    .map(|v| clamp(v, &mut clamped))
                    .collect(),
            };
            let qr: Vec<f32> = match &qr_p {
                None => vec![0.0; nz],
                Some(src) => remap(src, Extrap::Constant, &mut scratch)
                    .into_iter()
                    .map(|v| clamp(v, &mut clamped))
                    .collect(),
            };

            // Vertical velocity lives on the layer interfaces, and its own top
            // and bottom are the child's rigid lid and ground: the reference
            // stream writes zero there and so does this.
            let mut wscratch = vec![0.0f32; pz + 1];
            for k in 0..=pz {
                wscratch[k] = op.apply_column(&w_p, k);
            }
            let mut w = vec![0.0f32; nzp1];
            for k in 1..nz {
                w[k] = vertical_interp(
                    child_metrics.zgrid[c][k],
                    src_iface,
                    &wscratch,
                    Extrap::Constant,
                )
                .unwrap_or(wscratch[pz]);
            }

            (qv, qc, qr, rho, theta, w, clamped)
        })
        .collect();
    receipt.negative_mixing_ratios_clamped = cells.iter().map(|c| c.6).sum();

    // The wind, per child edge.
    let u_child: Vec<Vec<f32>> = (0..n_edges)
        .into_par_iter()
        .map(|e| {
            let op = &transfer.edge_ops[e];
            let src = &transfer.edge_src_mid[e];
            let tgt = &transfer.edge_tgt_mid[e];
            let mut scratch = vec![0.0f32; pz];
            for k in 0..pz {
                scratch[k] = op.apply_column(&u_p, k);
            }
            (0..nz)
                .map(|k| {
                    vertical_interp(tgt[k], src, &scratch, Extrap::Linear)
                        .unwrap_or(scratch[pz - 1])
                })
                .collect()
        })
        .collect();

    // Flatten in file order.
    let mut lbc_qv = Vec::with_capacity(n_cells * nz);
    let mut lbc_qc = Vec::with_capacity(n_cells * nz);
    let mut lbc_qr = Vec::with_capacity(n_cells * nz);
    let mut lbc_rho = Vec::with_capacity(n_cells * nz);
    let mut lbc_theta = Vec::with_capacity(n_cells * nz);
    let mut lbc_w = Vec::with_capacity(n_cells * nzp1);
    for (qv, qc, qr, rho, theta, w, _) in &cells {
        lbc_qv.extend_from_slice(qv);
        lbc_qc.extend_from_slice(qc);
        lbc_qr.extend_from_slice(qr);
        lbc_rho.extend_from_slice(rho);
        lbc_theta.extend_from_slice(theta);
        lbc_w.extend_from_slice(w);
    }
    let mut lbc_u = Vec::with_capacity(n_edges * nz);
    for col in &u_child {
        lbc_u.extend_from_slice(col);
    }

    let time_seconds = seconds_between(&cfg.start_time, &interval.valid_time)?;
    receipt.time_seconds = time_seconds as f64;

    let out_path = cfg.out_dir.join(emit::lbc_file_name(&interval.valid_time));
    let fields = emit::LbcFields {
        qv: &lbc_qv,
        qc: &lbc_qc,
        qr: &lbc_qr,
        u: &lbc_u,
        w: &lbc_w,
        rho: &lbc_rho,
        theta: &lbc_theta,
    };
    emit::write_lbc_file(
        &out_path,
        lbc_cfg,
        header,
        n_cells,
        n_edges,
        nz,
        &interval.valid_time,
        time_seconds as f32,
        &fields,
    )?;
    receipt.out_path = out_path.display().to_string();
    receipt.out_sha256 = crate::sha256_file(&out_path)?;
    receipt.seconds = started.elapsed().as_secs_f64();
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lbc::source::Registry;

    #[test]
    fn a_first_guess_row_is_refused_on_the_parent_native_route() {
        let cfg = ParentConfig {
            grid_path: PathBuf::from("nowhere.nc"),
            parent_grid: None,
            out_dir: PathBuf::from("nowhere"),
            start_time: "2026-08-12_06:00:00".to_string(),
            stop_time: "2026-08-12_12:00:00".to_string(),
            intervals: vec![BoundaryInterval {
                valid_time: "2026-08-12_06:00:00".to_string(),
                met_path: PathBuf::from("nowhere"),
            }],
            fg_interval_seconds: 10800,
            source_row: crate::lbc::source::DEFAULT_ROW.to_string(),
            registry_path: None,
            attr_overrides: Default::default(),
            provenance: String::new(),
            without_snap: false,
        };
        let err = build_from_parent(&cfg).unwrap_err().to_string();
        assert!(err.contains("FirstGuess"), "{err}");
        assert!(err.contains("rebuilds the boundary state"), "{err}");
    }

    #[test]
    fn no_intervals_is_refused_before_anything_opens() {
        let cfg = ParentConfig {
            grid_path: PathBuf::from("nowhere.nc"),
            parent_grid: None,
            out_dir: PathBuf::from("nowhere"),
            start_time: "2026-08-12_06:00:00".to_string(),
            stop_time: "2026-08-12_12:00:00".to_string(),
            intervals: Vec::new(),
            fg_interval_seconds: 10800,
            source_row: "unstructured-native-stream".to_string(),
            registry_path: None,
            attr_overrides: Default::default(),
            provenance: String::new(),
            without_snap: false,
        };
        let err = build_from_parent(&cfg).unwrap_err().to_string();
        assert!(err.contains("nothing to produce"), "{err}");
    }

    #[test]
    fn a_missing_parent_frame_names_what_it_means() {
        let err = RoleReader::open(Path::new("a-frame-that-was-never-written.nc"), None)
            .err()
            .expect("a frame that does not exist must be refused")
            .to_string();
        assert!(err.contains("never reached this boundary time"), "{err}");
    }

    #[test]
    fn a_transposed_column_count_is_refused_rather_than_reshaped() {
        let err = split_columns(vec![0.0; 10], 3, 4, "zgrid").unwrap_err().to_string();
        assert!(err.contains("transpose"), "{err}");
    }

    #[test]
    fn the_registry_reaches_the_route_by_name_only() {
        // Nothing in this module mentions a model.  The only way a source
        // enters is a row name, and an unknown one is refused with the
        // registry listed.
        let reg = Registry::built_in().unwrap();
        assert!(reg.row("unstructured-port-stream").is_ok());
        assert!(reg.row("something-nobody-added").is_err());
    }
}
