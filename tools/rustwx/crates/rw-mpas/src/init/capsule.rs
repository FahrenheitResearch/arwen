//! The mesh, the statics, and the vertical-metric capsule.
//!
//! The front door produces this file contract either from a supplied native
//! capsule or from Rust vertical/geometry builders. The assertion compares
//! zgrid with the declared reference contract, which may be the same Rust
//! artifact. The front-door receipt identifies its producer; reading this
//! contract alone does not establish native identity.

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

/// The vertical metric block, read from the front-door artifact contract.
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

/// `file.read_array_f64(name)?.into_values()`, read as slabs along the
/// outermost axis longer than one on rayon workers.  The decoder is serial
/// per call, and the vertical metrics alone are ~450 M words on the 30 km
/// mesh.  Slabs are joined in order, so these are the same words in the same
/// order; anything that does not split cleanly takes the one-call read.
pub(crate) fn read_f64_par(file: &netcrust::File, name: &str) -> netcrust::Result<Vec<f64>> {
    use netcrust::{NcSliceInfo, NcSliceInfoElem};
    use rayon::prelude::*;
    const SLAB_WORDS: usize = 1 << 22;
    let Some(variable) = file.variable(name) else {
        return Ok(file.read_array_f64(name)?.into_values());
    };
    let shape: Vec<usize> = variable.dimensions().iter().map(|d| d.len()).collect();
    let total: usize = shape.iter().product();
    let axis = shape.iter().position(|&n| n > 1);
    let (Some(axis), false, true) = (axis, shape.contains(&0), total >= 2 * SLAB_WORDS) else {
        return Ok(file.read_array_f64(name)?.into_values());
    };
    let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
    let step = (SLAB_WORDS / inner).max(1);
    let mut out = vec![0.0f64; total];
    out.par_chunks_mut(step * inner)
        .enumerate()
        .try_for_each(|(i, dst)| -> netcrust::Result<()> {
            let start = i * step;
            let selection = NcSliceInfo {
                selections: shape
                    .iter()
                    .enumerate()
                    .map(|(d, &n)| NcSliceInfoElem::Slice {
                        start: if d == axis { start as u64 } else { 0 },
                        end: if d == axis { (start + step).min(n) as u64 } else { n as u64 },
                        step: 1,
                    })
                    .collect(),
            };
            let words = file.read_array_f64_slice(name, &selection)?.into_values();
            if words.len() != dst.len() {
                return Err(netcrust::Error::Hdf5(format!(
                    "{name}: slab {i} holds {} word(s), expected {}",
                    words.len(),
                    dst.len()
                )));
            }
            dst.copy_from_slice(&words);
            Ok(())
        })?;
    Ok(out)
}

fn read_f32(file: &netcrust::File, name: &str) -> MpasResult<Vec<f32>> {
    use rayon::prelude::*;
    let values = read_f64_par(file, name)
        .map_err(|e| MpasError::Refusal(format!("capsule has no readable {name}: {e}")))?;
    Ok(values.into_par_iter().map(|v| v as f32).collect())
}

fn read_i32(file: &netcrust::File, name: &str) -> MpasResult<Vec<i32>> {
    let values = read_f64_par(file, name)
        .map_err(|e| MpasError::Refusal(format!("capsule has no readable {name}: {e}")))?;
    use rayon::prelude::*;
    Ok(values
        .into_par_iter()
        .map(|v| v.round() as i32)
        .collect())
}

/// Inspect stream declarations without reading any weather-field payload.
pub fn requested_fields(path:&Path,names:&[&str])->MpasResult<Vec<String>> {
    let file=netcrust::File::open(path)?;
    Ok(names.iter().filter(|name|file.variable(name).is_some()).map(|name|name.to_string()).collect())
}

/// Read the optional met-stage diagnostic contract. Missing geometry when
/// diagnostics are requested would otherwise leave capsule winds in the output.
pub fn read_reconstruction_contract(path:&Path,nc:usize,me:usize)
    -> MpasResult<Option<(Vec<String>,Vec<f32>,bool)>> {
    let file=netcrust::File::open(path)?;
    let names:Vec<String>=super::reconstruction::FIELDS.into_iter()
        .filter(|name|file.variable(name).is_some()).map(str::to_string).collect();
    if names.is_empty() {return Ok(None);}
    let coefficients=read_f32(&file,"coeffs_reconstruct")?;
    if coefficients.len()!=nc*me*3 {
        return Err(MpasError::Refusal("coeffs_reconstruct dimensions differ from the wind reconstruction stencil".into()));
    }
    let sphere=file.attribute("on_a_sphere").and_then(|a|a.as_string().map(str::to_string))
        .ok_or_else(||MpasError::Refusal("wind diagnostic rotation requires the declared on_a_sphere attribute".into()))?;
    let sphere=match sphere.trim().to_lowercase().as_str() {
        "yes"|"true"|".true."|"1"=>true,
        "no"|"false"|".false."|"0"=>false,
        _=>return Err(MpasError::Refusal(format!("unknown on_a_sphere value {sphere:?}; diagnostic rotation would be ambiguous"))),
    };
    Ok(Some((names,coefficients,sphere)))
}

fn dim(file: &netcrust::File, name: &str) -> MpasResult<usize> {
    file.dimension(name)
        .map(|d| d.len())
        .ok_or_else(|| MpasError::Refusal(format!("file has no {name} dimension")))
}

/// `(nCells, nEdges)` from a file's header alone, so the readers that need
/// them can start without waiting for the mesh arrays.
pub fn mesh_sizes(path: &Path) -> MpasResult<(usize, usize)> {
    let file = netcrust::File::open(path)?;
    Ok((dim(&file, "nCells")?, dim(&file, "nEdges")?))
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
    use rayon::prelude::*;
    let cells_on_edge: Vec<[usize; 2]> = (0..n_edges)
        .into_par_iter()
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
        .into_par_iter()
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

    use rayon::prelude::*;
    let per_cell = |flat: Vec<f32>, levels: usize, name: &str| -> MpasResult<Vec<Vec<f32>>> {
        if flat.len() != levels * n_cells {
            return per_cell(flat, levels, name);
        }
        Ok(flat.par_chunks(levels).map(<[f32]>::to_vec).collect())
    };

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
            .into_par_iter()
            .map(|e| {
                let base = e * 2 * nzp1;
                [
                    flat[base..base + nzp1].to_vec(),
                    flat[base + nzp1..base + 2 * nzp1].to_vec(),
                ]
            })
            .collect())
    };

    // The four big arrays are independent reads; take their results in the
    // order the serial reader used so a failure names the same array.
    let ((zgrid, zz), (zb, zb3)) = rayon::join(
        || {
            rayon::join(
                || per_cell(read_f32(&file, "zgrid")?, nzp1, "zgrid"),
                || per_cell(read_f32(&file, "zz")?, nz, "zz"),
            )
        },
        || rayon::join(|| read_edge_pair("zb"), || read_edge_pair("zb3")),
    );
    let (zgrid, zz) = (zgrid?, zz?);
    let (fzm, fzp, dzu, rdzw) = (
        read_f32(&file, "fzm")?,
        read_f32(&file, "fzp")?,
        read_f32(&file, "dzu")?,
        read_f32(&file, "rdzw")?,
    );
    Ok(VerticalMetrics {
        n_vert_levels: nz,
        zgrid,
        zz,
        fzm,
        fzp,
        dzu,
        rdzw,
        zb: zb?,
        zb3: zb3?,
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
