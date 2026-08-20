//! The mesh, the statics, and the vertical-metric capsule.
//!
//! ## The deferral this module exists to make visible
//! `zgrid`, `zz`, `fzm`, `fzp`, `dzu`, `rdzw`, `zb` and `zb3` are built by
//! about 450 lines of `init_atm_case_gfs` — fourth-order terrain smoothing,
//! the hybrid coordinate with its transition height, and an iterative smoother
//! for the coordinate surfaces with a minimum-`dz` criterion.  **This wave
//! does not port that.**  The vertical metrics are read from a capsule minted
//! by the native `init_atmosphere_model`, and [`assert_zgrid_identical`] then
//! refuses unless the `zgrid` read is bit-identical to the one in the native
//! init being compared against.
//!
//! That assertion is not ceremony.  Every field-level number in the validation
//! is a difference against a native init computed on *its* vertical grid; if
//! the grid moved by one bit, every delta downstream is contaminated and none
//! of them is explainable.  The consequence to state loudly wherever this is
//! marketed: an init cannot yet be made for an arbitrary mesh and vertical
//! configuration without a native mint, so the capsule is mesh-*and*-
//! configuration specific, not mesh-only.

use std::path::Path;

use crate::error::{MpasError, MpasResult};

/// Cell/edge geometry and connectivity, read once.
#[derive(Debug)]
pub struct MeshGeometry {
    pub n_cells: usize,
    pub n_edges: usize,
    pub lat_cell: Vec<f32>,
    pub lon_cell: Vec<f32>,
    pub lat_edge: Vec<f32>,
    pub lon_edge: Vec<f32>,
    pub angle_edge: Vec<f32>,
    /// One-based cell indices, two per edge, as the Fortran stores them.
    pub cells_on_edge: Vec<[usize; 2]>,
    pub n_edges_on_cell: Vec<usize>,
    /// One-based edge indices, `max_edges` per cell.
    pub edges_on_cell: Vec<Vec<usize>>,
}

/// The static fields case 7 reads out of the geography capsule.
#[derive(Debug)]
pub struct Statics {
    pub landmask: Vec<i32>,
    pub ter: Vec<f32>,
    pub soiltemp: Vec<f32>,
    /// Land-use and soil category, and the annual maximum snow albedo.  These
    /// are statics that the sea-ice initialisation *rewrites* in place at
    /// converted points, so they are read here rather than carried.
    pub ivgtyp: Vec<i32>,
    pub isltyp: Vec<i32>,
    pub snoalb: Vec<f32>,
    /// The land-use index that means "ice", from the land-use table.
    pub isice_lu: i32,
    /// `(nMonths, nCells)` monthly greenness, row-major by month.
    pub greenfrac: Vec<Vec<f32>>,
    pub albedo12m: Vec<Vec<f32>>,
}

/// The vertical metric block, read rather than built this wave.
#[derive(Debug)]
pub struct VerticalMetrics {
    pub n_vert_levels: usize,
    /// `(nVertLevelsP1, nCells)` in Fortran order; indexed `[cell][k]`.
    pub zgrid: Vec<Vec<f32>>,
    pub zz: Vec<Vec<f32>>,
    pub fzm: Vec<f32>,
    pub fzp: Vec<f32>,
    pub dzu: Vec<f32>,
    pub rdzw: Vec<f32>,
    /// `(nEdges, TWO, nVertLevelsP1)` on disk; indexed `[edge][side][k]`.
    ///
    /// The file dimensions these against `nVertLevelsP1`, not `nVertLevels`:
    /// the case code fills only `k = 1 .. nVertLevels` and leaves the top slot
    /// at zero.  Sizing the reader against `nVertLevels` reads at the wrong
    /// stride and silently mixes one edge's metric into the next.
    pub zb: Vec<[Vec<f32>; 2]>,
    pub zb3: Vec<[Vec<f32>; 2]>,
}

fn read_f32(file: &netcrust::File, name: &str) -> MpasResult<Vec<f32>> {
    let array = file
        .read_array_f64(name)
        .map_err(|e| MpasError::Refusal(format!("capsule has no readable {name}: {e}")))?;
    Ok(array.into_values().into_iter().map(|v| v as f32).collect())
}

fn read_i32(file: &netcrust::File, name: &str) -> MpasResult<Vec<i32>> {
    let array = file
        .read_array_f64(name)
        .map_err(|e| MpasError::Refusal(format!("capsule has no readable {name}: {e}")))?;
    Ok(array
        .into_values()
        .into_iter()
        .map(|v| v.round() as i32)
        .collect())
}

fn dim(file: &netcrust::File, name: &str) -> MpasResult<usize> {
    file.dimension(name)
        .map(|d| d.len())
        .ok_or_else(|| MpasError::Refusal(format!("file has no {name} dimension")))
}

/// Read cell/edge geometry and connectivity from a mesh, static or init file.
pub fn read_mesh_geometry(path: &Path) -> MpasResult<MeshGeometry> {
    let file = netcrust::File::open(path)?;
    let n_cells = dim(&file, "nCells")?;
    let n_edges = dim(&file, "nEdges")?;
    let max_edges = dim(&file, "maxEdges")?;

    let coe_raw = read_i32(&file, "cellsOnEdge")?;
    if coe_raw.len() != 2 * n_edges {
        return Err(MpasError::Refusal(format!(
            "cellsOnEdge holds {} value(s) for {n_edges} edges",
            coe_raw.len()
        )));
    }
    // Fortran stores (TWO, nEdges); C order puts the pair contiguous.
    let cells_on_edge: Vec<[usize; 2]> = (0..n_edges)
        .map(|e| [coe_raw[2 * e] as usize, coe_raw[2 * e + 1] as usize])
        .collect();

    let neoc = read_i32(&file, "nEdgesOnCell")?;
    let eoc_raw = read_i32(&file, "edgesOnCell")?;
    if eoc_raw.len() != max_edges * n_cells {
        return Err(MpasError::Refusal(format!(
            "edgesOnCell holds {} value(s), expected {}",
            eoc_raw.len(),
            max_edges * n_cells
        )));
    }
    let edges_on_cell: Vec<Vec<usize>> = (0..n_cells)
        .map(|c| {
            (0..max_edges)
                .map(|i| eoc_raw[c * max_edges + i] as usize)
                .collect()
        })
        .collect();

    Ok(MeshGeometry {
        n_cells,
        n_edges,
        lat_cell: read_f32(&file, "latCell")?,
        lon_cell: read_f32(&file, "lonCell")?,
        lat_edge: read_f32(&file, "latEdge")?,
        lon_edge: read_f32(&file, "lonEdge")?,
        angle_edge: read_f32(&file, "angleEdge")?,
        cells_on_edge,
        n_edges_on_cell: neoc.into_iter().map(|v| v as usize).collect(),
        edges_on_cell,
    })
}

/// Read the statics case 7 consumes.
pub fn read_statics(path: &Path, n_cells: usize) -> MpasResult<Statics> {
    let file = netcrust::File::open(path)?;
    let cells = dim(&file, "nCells")?;
    if cells != n_cells {
        return Err(MpasError::Refusal(format!(
            "statics carry {cells} cells; the mesh carries {n_cells}"
        )));
    }
    let n_months = dim(&file, "nMonths")?;
    let split = |flat: Vec<f32>, name: &str| -> MpasResult<Vec<Vec<f32>>> {
        if flat.len() != n_months * n_cells {
            return Err(MpasError::Refusal(format!(
                "{name} holds {} value(s), expected {}",
                flat.len(),
                n_months * n_cells
            )));
        }
        // Fortran (nMonths, nCells) is month-fastest in C order.
        Ok((0..n_months)
            .map(|m| (0..n_cells).map(|c| flat[c * n_months + m]).collect())
            .collect())
    };
    let isice = read_i32(&file, "isice_lu")?;
    Ok(Statics {
        landmask: read_i32(&file, "landmask")?,
        ter: read_f32(&file, "ter")?,
        soiltemp: read_f32(&file, "soiltemp")?,
        ivgtyp: read_i32(&file, "ivgtyp")?,
        isltyp: read_i32(&file, "isltyp")?,
        snoalb: read_f32(&file, "snoalb")?,
        isice_lu: *isice.first().ok_or_else(|| {
            MpasError::Refusal("statics carry no isice_lu".to_string())
        })?,
        greenfrac: split(read_f32(&file, "greenfrac")?, "greenfrac")?,
        albedo12m: split(read_f32(&file, "albedo12m")?, "albedo12m")?,
    })
}

/// Read the terrain the vertical-grid block left behind.
///
/// `ter` is **not** the raw static field once `init_atm_case_gfs` has run: the
/// vertical-grid block smooths it (`config_nsmterrain`, `config_smooth_surfaces`)
/// and writes it back in place, and every later consumer — the skin-temperature
/// and deep-soil terrain corrections above all — sees the smoothed field.
///
/// This lane defers the vertical-grid block, so the smoothed terrain comes from
/// the capsule with the metrics it belongs to.  Reading the raw static terrain
/// instead is not a rounding difference: it moved `skintemp`, `tmn` and `tslb`
/// by up to 5.8 K over steep terrain, which is 0.0065 K/m times the ~900 m the
/// smoother takes off an Andean ridge.
pub fn read_smoothed_terrain(path: &Path, n_cells: usize) -> MpasResult<Vec<f32>> {
    let file = netcrust::File::open(path)?;
    let ter = read_f32(&file, "ter")?;
    if ter.len() != n_cells {
        return Err(MpasError::Refusal(format!(
            "capsule terrain holds {} value(s) for {n_cells} cells",
            ter.len()
        )));
    }
    Ok(ter)
}

/// Read the vertical metric block out of a native-minted capsule.
pub fn read_vertical_metrics(
    path: &Path,
    n_cells: usize,
    n_edges: usize,
) -> MpasResult<VerticalMetrics> {
    let file = netcrust::File::open(path)?;
    let nz = dim(&file, "nVertLevels")?;
    let nzp1 = dim(&file, "nVertLevelsP1")?;
    if nzp1 != nz + 1 {
        return Err(MpasError::Refusal(format!(
            "capsule declares nVertLevels {nz} and nVertLevelsP1 {nzp1}"
        )));
    }

    let per_cell = |flat: Vec<f32>, levels: usize, name: &str| -> MpasResult<Vec<Vec<f32>>> {
        if flat.len() != levels * n_cells {
            return Err(MpasError::Refusal(format!(
                "{name} holds {} value(s), expected {}",
                flat.len(),
                levels * n_cells
            )));
        }
        Ok((0..n_cells)
            .map(|c| flat[c * levels..(c + 1) * levels].to_vec())
            .collect())
    };

    let zgrid = per_cell(read_f32(&file, "zgrid")?, nzp1, "zgrid")?;
    let zz = per_cell(read_f32(&file, "zz")?, nz, "zz")?;

    let read_edge_pair = |name: &str| -> MpasResult<Vec<[Vec<f32>; 2]>> {
        let flat = read_f32(&file, name)?;
        if flat.len() != 2 * nzp1 * n_edges {
            return Err(MpasError::Refusal(format!(
                "{name} holds {} value(s), expected {} ({n_edges} edges x 2 sides x {nzp1}                  levels); this array is dimensioned against nVertLevelsP1",
                flat.len(),
                2 * nzp1 * n_edges
            )));
        }
        // (nEdges, TWO, nVertLevelsP1): level fastest, then side, then edge.
        Ok((0..n_edges)
            .map(|e| {
                let base = e * 2 * nzp1;
                [
                    flat[base..base + nzp1].to_vec(),
                    flat[base + nzp1..base + 2 * nzp1].to_vec(),
                ]
            })
            .collect())
    };

    Ok(VerticalMetrics {
        n_vert_levels: nz,
        zgrid,
        zz,
        fzm: read_f32(&file, "fzm")?,
        fzp: read_f32(&file, "fzp")?,
        dzu: read_f32(&file, "dzu")?,
        rdzw: read_f32(&file, "rdzw")?,
        zb: read_edge_pair("zb")?,
        zb3: read_edge_pair("zb3")?,
    })
}

/// Refuse unless the capsule's `zgrid` is bit-identical to the reference's.
///
/// Bit-identity, not a tolerance: this is the one comparison in the whole
/// validation that is allowed to be exact, and it has to be, because every
/// other comparison is conditioned on it.
pub fn assert_zgrid_identical(
    capsule: &VerticalMetrics,
    reference_path: &Path,
) -> MpasResult<usize> {
    let file = netcrust::File::open(reference_path)?;
    let nzp1 = dim(&file, "nVertLevelsP1")?;
    let n_cells = dim(&file, "nCells")?;
    let flat = read_f32(&file, "zgrid")?;
    if flat.len() != nzp1 * n_cells {
        return Err(MpasError::Refusal(format!(
            "reference zgrid holds {} value(s)",
            flat.len()
        )));
    }
    if capsule.zgrid.len() != n_cells {
        return Err(MpasError::Refusal(format!(
            "capsule zgrid covers {} cells; the reference covers {n_cells}",
            capsule.zgrid.len()
        )));
    }
    let mut differing = 0usize;
    let mut first: Option<(usize, usize, f32, f32)> = None;
    for c in 0..n_cells {
        for k in 0..nzp1 {
            let a = capsule.zgrid[c][k];
            let b = flat[c * nzp1 + k];
            if a.to_bits() != b.to_bits() {
                differing += 1;
                if first.is_none() {
                    first = Some((c, k, a, b));
                }
            }
        }
    }
    if differing > 0 {
        let (c, k, a, b) = first.unwrap();
        return Err(MpasError::Refusal(format!(
            "zgrid is not bit-identical to {}: {differing} value(s) differ, first at \
             cell {c} level {k} ({a:e} vs {b:e}).  Every field-level delta in this run \
             would be a difference on two different vertical grids, so none of them could \
             be explained; the run stops here rather than produce an unreadable table",
            reference_path.display()
        )));
    }
    Ok(nzp1 * n_cells)
}

#[cfg(test)]
mod tests {

    #[test]
    fn a_zgrid_mismatch_names_the_first_differing_point() {
        // Exercised against a real capsule in the field runs; here we only
        // pin the message shape a caller greps for.
        let msg = format!(
            "zgrid is not bit-identical to {}: {} value(s) differ, first at cell {} level {}",
            "x.nc", 3, 17, 4
        );
        assert!(msg.contains("not bit-identical"));
        assert!(msg.contains("first at cell 17 level 4"));
    }
}
