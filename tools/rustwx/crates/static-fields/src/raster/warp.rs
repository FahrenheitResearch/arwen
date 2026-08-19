//! LANE 3.  Mosaic + reprojection.
//!
//! Defined behaviour (the spec the parity tolerance gates against):
//!
//! * **mosaic**: rasterio-`merge` grid arithmetic (output width/height
//!   = `round((extent)/res)`, origin at the declared west/north), then
//!   first-writer-wins painting in tile list order with nearest
//!   sampling of each source at the output pixel centre — no
//!   elevation is invented; coarser latitude bands replicate, and the
//!   area-average to the model grid is what actually reduces them
//!   (the `derive_global_terrain_window` contract).  Cells no tile
//!   reaches, cells masked in every reaching tile, and cells equal to
//!   the declared in-band void sentinel stay NaN and are counted; the
//!   derive layer decides the fill.  One measured divergence from
//!   rasterio: where adjacent tiles are staggered by a sub-pixel
//!   offset, rasterio's integer window alignment can drop a one-pixel
//!   seam column to nodata even though a source pixel contains the
//!   output centre; this mosaic samples by centre containment and
//!   keeps it, so its hole set is a SUBSET of rasterio's (asserted by
//!   the parity harness on a pinned real-tile seam);
//! * **area-average warp**: GDAL `Resampling.average`'s own kernel
//!   shape (GWKAverageOrMode) — for every destination cell, the
//!   bounding box of its projected TOP-LEFT and BOTTOM-RIGHT corners
//!   in source pixel space, expanded to whole pixels by
//!   `floor(min + 1e-10) .. ceil(max - 1e-10)`, and the equal-weight
//!   mean of every VALID source pixel in that rectangle (an
//!   area-intersection rule, so a destination cell finer than the
//!   source still averages >= 1 pixel).  A destination cell whose
//!   rectangle holds no valid pixel falls back, in order: bilinear
//!   sample at the cell centre when all four neighbours are valid;
//!   the containing source pixel when valid; else NaN.  GDAL remains
//!   a black box (approximate transformers, chunked edge handling),
//!   so the harness measures the residual against rasterio on pinned
//!   footprints and gates on the recorded caps;
//! * **category fractions**: per-category counting over the same
//!   corner-box kernel, normalized by the box's valid total — the
//!   `_resample_category_array` contract, including the coverage
//!   discipline (unreached cells are NaN so the caller's coverage
//!   gates fire exactly as the Python's do; unreached cells with a
//!   valid containing pixel take its category at fraction 1);
//! * **nearest / bilinear**: standard pull-based, for the declared
//!   method names.
//!
//! Parallelism: rayon over destination rows; every accumulation is
//! per-cell in fixed source scan order — bit-stable run to run and
//! equal to the serial result by construction.

use rayon::prelude::*;

use crate::error::{Result, StaticError};
use crate::raster::{Crs, Raster};
use crate::types::{Grid2, Stack3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resampling {
    Average,
    Bilinear,
    Nearest,
}

impl Resampling {
    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "average" => Resampling::Average,
            "bilinear" => Resampling::Bilinear,
            "nearest" => Resampling::Nearest,
            other => {
                return Err(StaticError::Invalid(format!(
                    "unsupported continuous resampling method {other:?}"
                )))
            }
        })
    }
}

/// Rows per rayon work block in the mosaic paint.
const ROW_BLOCK: usize = 256;

// ---------------------------------------------------------------------------
// Mosaic
// ---------------------------------------------------------------------------

/// Mosaic tiles onto a uniform grid over `bounds = [west, south, east,
/// north]`.  `resolution` `None` inherits the first tile's pixel size
/// (the staged-tile contract, where all tiles share one grid);
/// `Some(r)` declares a square output resolution (the latitude-banded
/// contract).  Returns the NaN-holed mosaic and the hole count after
/// `source_nodata` masking; the caller decides the fill.
pub fn mosaic(
    tiles: &[Raster],
    bounds: [f64; 4],
    resolution: Option<f64>,
    source_nodata: Option<f64>,
) -> Result<(Raster, usize)> {
    if tiles.is_empty() {
        return Err(StaticError::Invalid(
            "terrain window derivation requires >= 1 tile".into(),
        ));
    }
    let crs = tiles[0].crs.clone();
    for tile in tiles {
        if tile.crs != crs {
            return Err(StaticError::Invalid(
                "mosaic tiles disagree on CRS".into(),
            ));
        }
        let t = &tile.transform;
        if t[1] != 0.0 || t[3] != 0.0 || t[0] <= 0.0 || t[4] >= 0.0 {
            return Err(StaticError::Invalid(format!(
                "mosaic tile transform {t:?} is not north-up rectilinear"
            )));
        }
    }
    let (res_x, res_y) = match resolution {
        Some(r) => (r, r),
        None => (tiles[0].transform[0], -tiles[0].transform[4]),
    };
    let [west, south, east, north] = bounds;
    // rasterio.merge: output covers the bounds completely.
    let out_w = ((east - west) / res_x).round() as i64;
    let out_h = ((north - south) / res_y).round() as i64;
    if out_w <= 0 || out_h <= 0 {
        return Err(StaticError::Invalid(format!(
            "mosaic bounds {bounds:?} at resolution {res_x}x{res_y} \
             yield an empty grid"
        )));
    }
    let (out_w, out_h) = (out_w as usize, out_h as usize);
    let transform = [res_x, 0.0, west, 0.0, -res_y, north];

    let values: Vec<f64> = (0..out_h)
        .collect::<Vec<_>>()
        .par_chunks(ROW_BLOCK)
        .map(|rows| {
            let mut block = vec![f64::NAN; rows.len() * out_w];
            for (block_row, row) in rows.iter().enumerate() {
                let y = north - (*row as f64 + 0.5) * res_y;
                for col in 0..out_w {
                    let x = west + (col as f64 + 0.5) * res_x;
                    let slot = &mut block[block_row * out_w + col];
                    for tile in tiles {
                        let t = &tile.transform;
                        let src_col = ((x - t[2]) / t[0]).floor();
                        let src_row = ((y - t[5]) / t[4]).floor();
                        if src_col < 0.0
                            || src_row < 0.0
                            || src_col >= tile.nx as f64
                            || src_row >= tile.ny as f64
                        {
                            continue;
                        }
                        let value = tile.values
                            [src_row as usize * tile.nx + src_col as usize];
                        if value.is_nan() {
                            continue;
                        }
                        *slot = value;
                        break; // first-writer-wins, rasterio's default
                    }
                }
            }
            block
        })
        .collect::<Vec<_>>()
        .concat();

    let mut values = values;
    if let Some(sentinel) = source_nodata {
        for value in values.iter_mut() {
            if *value == sentinel {
                *value = f64::NAN;
            }
        }
    }
    let holes = values.iter().filter(|v| v.is_nan()).count();
    Ok((
        Raster { ny: out_h, nx: out_w, values, transform, crs },
        holes,
    ))
}

// ---------------------------------------------------------------------------
// The corner grid: destination pixel corners in source pixel space
// ---------------------------------------------------------------------------

/// Fractional source-pixel coordinates of every destination pixel
/// CORNER: two `(ny+1) x (nx+1)` row-major planes `(cols, rows)`.
/// NaN where the transform has no answer.
fn corner_grid(
    source: &Raster,
    dst_crs: &Crs,
    dst_transform: &[f64; 6],
    dst_ny: usize,
    dst_nx: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let inv_dst = dst_crs.point_projection()?;
    let fwd_src = source.crs.point_projection()?;
    let width = dst_nx + 1;
    let height = dst_ny + 1;
    let t = source.transform;
    let mut cols = vec![f64::NAN; width * height];
    let mut rows = vec![f64::NAN; width * height];
    cols.par_chunks_mut(width)
        .zip(rows.par_chunks_mut(width))
        .enumerate()
        .for_each(|(row, (col_row, row_row))| {
            let y = dst_transform[5] + dst_transform[4] * row as f64;
            for col in 0..width {
                let x = dst_transform[2] + dst_transform[0] * col as f64;
                let (lon, lat) = inv_dst.inverse(x, y);
                let (sx, sy) = fwd_src.forward(lon, lat);
                col_row[col] = (sx - t[2]) / t[0];
                row_row[col] = (sy - t[5]) / t[4];
            }
        });
    Ok((cols, rows))
}

/// GDAL's integer source-pixel span for one box edge pair:
/// `floor(lo + 1e-10) .. ceil(hi - 1e-10)` clamped to `0..len`,
/// returned as an inclusive `(first, last)`; `None` when empty.
#[inline]
fn gdal_span(lo: f64, hi: f64, len: usize) -> Option<(usize, usize)> {
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    let first = (lo + 1.0e-10).floor().max(0.0);
    let last_exclusive = (hi - 1.0e-10).ceil().min(len as f64);
    if first >= last_exclusive || last_exclusive <= 0.0 {
        return None;
    }
    Some((first as usize, last_exclusive as usize - 1))
}

/// Bilinear sample of `source` at fractional pixel-centre coordinates;
/// None unless all four neighbours are in-bounds and valid.
fn bilinear_sample(source: &Raster, col_f: f64, row_f: f64) -> Option<f64> {
    let px = col_f - 0.5;
    let py = row_f - 0.5;
    let i0 = px.floor();
    let j0 = py.floor();
    if i0 < 0.0
        || j0 < 0.0
        || i0 + 1.0 > source.nx as f64 - 1.0
        || j0 + 1.0 > source.ny as f64 - 1.0
    {
        return None;
    }
    let (i0, j0) = (i0 as usize, j0 as usize);
    let fx = px - i0 as f64;
    let fy = py - j0 as f64;
    let at = |j: usize, i: usize| source.values[j * source.nx + i];
    let v00 = at(j0, i0);
    let v01 = at(j0, i0 + 1);
    let v10 = at(j0 + 1, i0);
    let v11 = at(j0 + 1, i0 + 1);
    if v00.is_nan() || v01.is_nan() || v10.is_nan() || v11.is_nan() {
        return None;
    }
    Some(
        v00 * (1.0 - fx) * (1.0 - fy)
            + v01 * fx * (1.0 - fy)
            + v10 * (1.0 - fx) * fy
            + v11 * fx * fy,
    )
}

/// The containing source pixel at fractional pixel coordinates, if
/// in-bounds; `None` outside the grid.
fn containing_index(
    source_ny: usize,
    source_nx: usize,
    col_f: f64,
    row_f: f64,
) -> Option<usize> {
    let i = col_f.floor();
    let j = row_f.floor();
    if !i.is_finite()
        || !j.is_finite()
        || i < 0.0
        || j < 0.0
        || i >= source_nx as f64
        || j >= source_ny as f64
    {
        return None;
    }
    Some(j as usize * source_nx + i as usize)
}

/// The cell's GDAL box in source pixel space: the bounding box of the
/// projected TOP-LEFT and BOTTOM-RIGHT corners only (GWKAverageOrMode
/// transforms exactly this diagonal pair); `None` when either corner
/// failed to transform.
#[inline]
fn cell_box(
    cols: &[f64],
    rows: &[f64],
    width: usize,
    row: usize,
    col: usize,
) -> Option<(f64, f64, f64, f64)> {
    let tl = row * width + col;
    let br = (row + 1) * width + col + 1;
    let (c0, r0) = (cols[tl], rows[tl]);
    let (c1, r1) = (cols[br], rows[br]);
    if !c0.is_finite() || !r0.is_finite() || !c1.is_finite() || !r1.is_finite()
    {
        return None;
    }
    Some((c0.min(c1), c0.max(c1), r0.min(r1), r0.max(r1)))
}

// ---------------------------------------------------------------------------
// Continuous reprojection
// ---------------------------------------------------------------------------

/// Reproject one continuous raster onto the model mass grid
/// (south-north order on return, like `resample_continuous`).
pub fn reproject_continuous(
    source: &Raster,
    dst_crs: &Crs,
    dst_transform: [f64; 6],
    dst_ny: usize,
    dst_nx: usize,
    method: Resampling,
) -> Result<Grid2> {
    let (cols, rows) =
        corner_grid(source, dst_crs, &dst_transform, dst_ny, dst_nx)?;
    let width = dst_nx + 1;
    let mut north_first = vec![f64::NAN; dst_ny * dst_nx];

    north_first
        .par_chunks_mut(dst_nx)
        .enumerate()
        .for_each(|(row, out_row)| {
            for (col, slot) in out_row.iter_mut().enumerate() {
                // Cell centre in source pixel space, approximated by
                // the corner mean (curvature across one cell is far
                // below the parity tolerances).
                let centre_c = 0.25
                    * (cols[row * width + col]
                        + cols[row * width + col + 1]
                        + cols[(row + 1) * width + col]
                        + cols[(row + 1) * width + col + 1]);
                let centre_r = 0.25
                    * (rows[row * width + col]
                        + rows[row * width + col + 1]
                        + rows[(row + 1) * width + col]
                        + rows[(row + 1) * width + col + 1]);
                match method {
                    Resampling::Average => {
                        let reached = cell_box(&cols, &rows, width, row, col)
                            .and_then(|(min_c, max_c, min_r, max_r)| {
                                let (c0, c1) =
                                    gdal_span(min_c, max_c, source.nx)?;
                                let (r0, r1) =
                                    gdal_span(min_r, max_r, source.ny)?;
                                let mut sum = 0.0f64;
                                let mut count = 0u64;
                                for j in r0..=r1 {
                                    for i in c0..=c1 {
                                        let value =
                                            source.values[j * source.nx + i];
                                        if !value.is_nan() {
                                            sum += value;
                                            count += 1;
                                        }
                                    }
                                }
                                (count > 0).then(|| sum / count as f64)
                            });
                        *slot = reached
                            .or_else(|| {
                                bilinear_sample(source, centre_c, centre_r)
                            })
                            .or_else(|| {
                                containing_index(
                                    source.ny, source.nx, centre_c, centre_r,
                                )
                                .map(|at| source.values[at])
                                .filter(|value| !value.is_nan())
                            })
                            .unwrap_or(f64::NAN);
                    }
                    Resampling::Bilinear => {
                        *slot = bilinear_sample(source, centre_c, centre_r)
                            .unwrap_or(f64::NAN);
                    }
                    Resampling::Nearest => {
                        *slot = containing_index(
                            source.ny, source.nx, centre_c, centre_r,
                        )
                        .map(|at| source.values[at])
                        .unwrap_or(f64::NAN);
                    }
                }
            }
        });

    // Flip to south-north order.
    let mut data = vec![0.0f64; dst_ny * dst_nx];
    for row in 0..dst_ny {
        data[(dst_ny - 1 - row) * dst_nx..(dst_ny - row) * dst_nx]
            .copy_from_slice(&north_first[row * dst_nx..(row + 1) * dst_nx]);
    }
    Ok(Grid2 { ny: dst_ny, nx: dst_nx, data })
}

// ---------------------------------------------------------------------------
// Category fractions
// ---------------------------------------------------------------------------

/// Area fractions for already-classified pixels
/// (`_resample_category_array`): per-category counting over the
/// corner-box kernel + coverage + normalization, south-north order on
/// return.  `values`/`valid` are row-major over `source`'s grid;
/// `source.values` is only the georeference carrier here.
#[allow(clippy::too_many_arguments)]
pub fn reproject_category_fractions(
    values: &[i16],
    valid: &[bool],
    source: &Raster,
    dst_crs: &Crs,
    dst_transform: [f64; 6],
    dst_ny: usize,
    dst_nx: usize,
    category_count: usize,
) -> Result<Stack3> {
    if values.len() != source.ny * source.nx || valid.len() != values.len() {
        return Err(StaticError::Invalid(
            "category values and validity mask shapes differ".into(),
        ));
    }
    for (value, ok) in values.iter().zip(valid) {
        if *ok && (*value < 1 || *value as usize > category_count) {
            return Err(StaticError::Invalid(format!(
                "mapped category {value} is outside 1..{category_count}"
            )));
        }
    }
    let (cols, rows) =
        corner_grid(source, dst_crs, &dst_transform, dst_ny, dst_nx)?;
    let width = dst_nx + 1;
    let cells = dst_ny * dst_nx;

    // Row-parallel pull; each row writes its own pillar slice pattern,
    // gathered afterwards (plane-major assembly below).
    let per_row: Vec<Vec<f64>> = (0..dst_ny)
        .into_par_iter()
        .map(|row| {
            let mut out = vec![f64::NAN; category_count * dst_nx];
            let mut counts = vec![0u64; category_count];
            for col in 0..dst_nx {
                counts.iter_mut().for_each(|slot| *slot = 0);
                let mut total = 0u64;
                if let Some((min_c, max_c, min_r, max_r)) =
                    cell_box(&cols, &rows, width, row, col)
                {
                    if let (Some((c0, c1)), Some((r0, r1))) = (
                        gdal_span(min_c, max_c, source.nx),
                        gdal_span(min_r, max_r, source.ny),
                    ) {
                        for j in r0..=r1 {
                            for i in c0..=c1 {
                                let at = j * source.nx + i;
                                if valid[at] {
                                    counts[values[at] as usize - 1] += 1;
                                    total += 1;
                                }
                            }
                        }
                    }
                }
                if total > 0 {
                    for category in 0..category_count {
                        out[category * dst_nx + col] =
                            counts[category] as f64 / total as f64;
                    }
                    continue;
                }
                // Unreached: the containing valid source pixel (GDAL
                // average's upsampling limit), else NaN = uncovered.
                let centre_c = 0.25
                    * (cols[row * width + col]
                        + cols[row * width + col + 1]
                        + cols[(row + 1) * width + col]
                        + cols[(row + 1) * width + col + 1]);
                let centre_r = 0.25
                    * (rows[row * width + col]
                        + rows[row * width + col + 1]
                        + rows[(row + 1) * width + col]
                        + rows[(row + 1) * width + col + 1]);
                if let Some(at) = containing_index(
                    source.ny, source.nx, centre_c, centre_r,
                ) {
                    if valid[at] {
                        for category in 0..category_count {
                            out[category * dst_nx + col] =
                                if category as i16 + 1 == values[at] {
                                    1.0
                                } else {
                                    0.0
                                };
                        }
                    }
                }
            }
            out
        })
        .collect();

    // Assemble plane-major, south-north flipped.
    let mut data = vec![f64::NAN; category_count * cells];
    for (row, row_data) in per_row.iter().enumerate() {
        let flipped = dst_ny - 1 - row;
        for category in 0..category_count {
            let src = &row_data[category * dst_nx..(category + 1) * dst_nx];
            let dst = category * cells + flipped * dst_nx;
            data[dst..dst + dst_nx].copy_from_slice(src);
        }
    }
    Ok(Stack3 { planes: category_count, ny: dst_ny, nx: dst_nx, data })
}
