//! LANE 3.  Mosaic + reprojection.
//!
//! Defined behaviour (the spec the parity tolerance gates against):
//!
//! * **mosaic**: whole pixels of a fixed lattice covering the bounds
//!   (the first tile's own pixel grid when the resolution is inherited,
//!   pixel centres on whole multiples of a declared resolution), so every
//!   footprint cut from a source agrees on shared ground; this is
//!   rasterio-`merge` on the lattice-snapped bounds, the Python
//!   fallback's own arithmetic.  Then
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
                )));
            }
        })
    }
}

/// Rows per rayon work block in the mosaic paint.
const ROW_BLOCK: usize = 256;

// ---------------------------------------------------------------------------
// Mosaic
// ---------------------------------------------------------------------------

/// Mosaic tiles over `bounds = [west, south, east, north]` on the
/// source's terrain lattice ([`terrain_lattice`]).  `resolution` `None`
/// inherits the first tile's pixel grid (the staged-tile contract, where
/// all tiles share one grid); `Some(r)` declares a square output
/// resolution (the latitude-banded contract).  Returns the NaN-holed
/// mosaic and the hole count after `source_nodata` masking; the caller
/// decides the fill.
///
/// The output covers the bounds with whole lattice pixels.  It used to
/// start at each footprint's own west/north edge (rasterio's rule on the
/// raw bounds), which gave a moving nest's statics corridor and the nest
/// itself a different sub-pixel sampling of the same ground, so the
/// nest's first move was refused with "footprint-rebuilt statics differ".
pub fn mosaic(
    tiles: &[Raster],
    bounds: [f64; 4],
    resolution: Option<f64>,
    source_nodata: Option<f64>,
) -> Result<(Raster, usize)> {
    check_mosaic_tiles(tiles)?;
    let (pixel, origin) = terrain_lattice(tiles, resolution);
    mosaic_on_lattice(tiles, bounds, pixel, origin, source_nodata)
}

/// The fixed pixel lattice a terrain crop is cut on, `([res_x, res_y],
/// [origin_x, origin_y])`, with pixel edges at `origin + k * res`.
///
/// An inherited resolution keeps the first tile's own lattice, so each
/// output pixel IS a source pixel.  A declared resolution puts pixel
/// centres on whole multiples of that resolution, which is where the
/// point-sampled DEMs (GLO-30, SRTM) put their own samples; a lattice
/// with EDGES there would centre every output pixel on a source pixel
/// boundary, where the nearest pick is a tie.  `gpuwm.static.
/// highres_fetch._terrain_lattice` is the same rule for the fallback.
fn terrain_lattice(tiles: &[Raster], resolution: Option<f64>) -> ([f64; 2], [f64; 2]) {
    match resolution {
        Some(r) => ([r, r], [-0.5 * r, 0.5 * r]),
        None => {
            let t = tiles[0].transform;
            ([t[0], -t[4]], [t[2], t[5]])
        }
    }
}

/// Mosaic tiles onto a FIXED pixel lattice: pixel edges at
/// `origin + k * resolution` for whole `k`, covering `bounds = [west,
/// south, east, north]` with whole lattice pixels.
///
/// Two footprints cut from the same tiles on the same lattice sample
/// the same source pixel for the same ground and give it the same
/// georeferencing, so their shared ground is identical byte for byte.
/// Pixel centres are computed from their whole lattice index, never from
/// the crop's edge, so the same ground pixel gets the same coordinates
/// to the last bit in every crop.
pub fn mosaic_on_lattice(
    tiles: &[Raster],
    bounds: [f64; 4],
    resolution: [f64; 2],
    origin: [f64; 2],
    source_nodata: Option<f64>,
) -> Result<(Raster, usize)> {
    check_mosaic_tiles(tiles)?;
    let [res_x, res_y] = resolution;
    let [ox, oy] = origin;
    if !(res_x > 0.0 && res_y > 0.0) || !ox.is_finite() || !oy.is_finite() {
        return Err(StaticError::Invalid(format!(
            "mosaic lattice resolution {resolution:?} / origin {origin:?} \
             is not a positive finite lattice"
        )));
    }
    let [west, south, east, north] = bounds;
    // Columns count east from the origin, rows count south from it.
    let col0 = ((west - ox) / res_x).floor() as i64;
    let col1 = ((east - ox) / res_x).ceil() as i64;
    let row0 = ((oy - north) / res_y).floor() as i64;
    let row1 = ((oy - south) / res_y).ceil() as i64;
    paint_mosaic(
        tiles,
        bounds,
        [res_x, res_y],
        [ox, oy],
        [col0, row0],
        [col1 - col0, row1 - row0],
        source_nodata,
    )
}

fn check_mosaic_tiles(tiles: &[Raster]) -> Result<()> {
    if tiles.is_empty() {
        return Err(StaticError::Invalid(
            "terrain window derivation requires >= 1 tile".into(),
        ));
    }
    let crs = &tiles[0].crs;
    for tile in tiles {
        if &tile.crs != crs {
            return Err(StaticError::Invalid("mosaic tiles disagree on CRS".into()));
        }
        let t = &tile.transform;
        if t[1] != 0.0 || t[3] != 0.0 || t[0] <= 0.0 || t[4] >= 0.0 {
            return Err(StaticError::Invalid(format!(
                "mosaic tile transform {t:?} is not north-up rectilinear"
            )));
        }
    }
    Ok(())
}

/// Paint `size = [width, height]` output pixels whose centres are
/// `origin + (index + 0.5) * resolution` east and south, starting at
/// lattice index `first = [col, row]`.  First-writer-wins over the tiles
/// in list order, nearest sampling at the centre.
fn paint_mosaic(
    tiles: &[Raster],
    bounds: [f64; 4],
    resolution: [f64; 2],
    origin: [f64; 2],
    first: [i64; 2],
    size: [i64; 2],
    source_nodata: Option<f64>,
) -> Result<(Raster, usize)> {
    let [res_x, res_y] = resolution;
    let [ox, oy] = origin;
    let [col0, row0] = first;
    let [out_w, out_h] = size;
    if out_w <= 0 || out_h <= 0 {
        return Err(StaticError::Invalid(format!(
            "mosaic bounds {bounds:?} at resolution {res_x}x{res_y} \
             yield an empty grid"
        )));
    }
    let crs = tiles[0].crs.clone();
    let (out_w, out_h) = (out_w as usize, out_h as usize);
    let transform = [
        res_x,
        0.0,
        ox + col0 as f64 * res_x,
        0.0,
        -res_y,
        oy - row0 as f64 * res_y,
    ];

    let values: Vec<f64> = (0..out_h)
        .collect::<Vec<_>>()
        .par_chunks(ROW_BLOCK)
        .map(|rows| {
            let mut block = vec![f64::NAN; rows.len() * out_w];
            for (block_row, row) in rows.iter().enumerate() {
                let y = oy - ((row0 + *row as i64) as f64 + 0.5) * res_y;
                for col in 0..out_w {
                    let x = ox + ((col0 + col as i64) as f64 + 0.5) * res_x;
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
                        let value = tile.values[src_row as usize * tile.nx + src_col as usize];
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
        Raster {
            ny: out_h,
            nx: out_w,
            values,
            transform,
            crs,
        },
        holes,
    ))
}

/// Mosaic directly from source TIFF windows into deterministic output tiles.
/// Source payloads, mosaic pixels and compressed output are never held whole.
#[allow(clippy::too_many_arguments)]
pub fn mosaic_tiffs(
    paths: &[std::path::PathBuf],
    bounds: [f64; 4],
    resolution: Option<f64>,
    source_nodata: Option<f64>,
    out_path: &std::path::Path,
    output_nodata: Option<f64>,
    fill: Option<f64>,
) -> Result<(usize, usize, usize, Option<f64>)> {
    use super::geotiff::{self, TiffReader};
    let tiles: Vec<(Raster, Option<f64>)> = paths
        .iter()
        .map(|path| {
            let reader = TiffReader::open(path)?;
            Ok((
                Raster {
                    ny: reader.height,
                    nx: reader.width,
                    values: Vec::new(),
                    transform: reader.transform,
                    crs: reader.crs.unwrap_or(Crs::Geographic),
                },
                reader.nodata,
            ))
        })
        .collect::<Result<_>>()?;
    if tiles.is_empty() {
        return Err(StaticError::Invalid(
            "terrain window derivation requires >= 1 tile".into(),
        ));
    }
    let first_nodata = tiles[0].1;
    let crs = tiles[0].0.crs.clone();
    for (tile, _) in &tiles {
        let own = &tile.crs;
        let t = tile.transform;
        if own != &crs || t[1] != 0.0 || t[3] != 0.0 || t[0] <= 0.0 || t[4] >= 0.0 {
            return Err(StaticError::Invalid(
                "terrain tiles must share a north-up CRS".into(),
            ));
        }
    }
    let first = tiles[0].0.transform;
    let ([rx, ry], [ox, oy]) = match resolution {
        Some(r) => ([r, r], [-0.5 * r, 0.5 * r]),
        None => ([first[0], -first[4]], [first[2], first[5]]),
    };
    if !(rx > 0.0 && ry > 0.0) {
        return Err(StaticError::Invalid(
            "terrain mosaic resolution must be positive".into(),
        ));
    }
    let col0 = ((bounds[0] - ox) / rx).floor() as i64;
    let col1 = ((bounds[2] - ox) / rx).ceil() as i64;
    let row0 = ((oy - bounds[3]) / ry).floor() as i64;
    let row1 = ((oy - bounds[1]) / ry).ceil() as i64;
    if col1 <= col0 || row1 <= row0 {
        return Err(StaticError::Invalid(
            "terrain mosaic bounds yield an empty grid".into(),
        ));
    }
    let nx = (col1 - col0) as usize;
    let ny = (row1 - row0) as usize;
    let transform = [
        rx,
        0.0,
        ox + col0 as f64 * rx,
        0.0,
        -ry,
        oy - row0 as f64 * ry,
    ];
    let mut holes = 0usize;
    // At most eight open source files and 64 MiB of decoded block cache,
    // regardless of the number of terrain tiles in a domain.
    let mut readers: std::collections::VecDeque<(usize, TiffReader)> =
        std::collections::VecDeque::new();
    geotiff::write_band1_tiles(
        out_path,
        ny,
        nx,
        &transform,
        &crs,
        geotiff::SampleType::F32,
        output_nodata,
        |out_col, out_row, w, h| {
            let mut out = vec![f64::NAN; w * h];
            let x0 = ox + ((col0 + out_col as i64) as f64 + 0.5) * rx;
            let x1 = ox + ((col0 + (out_col + w - 1) as i64) as f64 + 0.5) * rx;
            let y0 = oy - ((row0 + out_row as i64) as f64 + 0.5) * ry;
            let y1 = oy - ((row0 + (out_row + h - 1) as i64) as f64 + 0.5) * ry;
            for (index, (tile, _)) in tiles.iter().enumerate() {
                let t = tile.transform;
                // A conservative pixel margin keeps containment rounding at
                // the far edge identical to the dense source-index rule.
                if x1 < t[2] - t[0].abs()
                    || x0 >= t[2] + tile.nx as f64 * t[0] + t[0].abs()
                    || y0 < t[5] + tile.ny as f64 * t[4] - t[4].abs()
                    || y1 > t[5] + t[4].abs()
                {
                    continue;
                }
                if let Some(position) = readers.iter().position(|(key, _)| *key == index) {
                    let entry = readers.remove(position).unwrap();
                    readers.push_back(entry);
                } else {
                    if readers.len() == 8 {
                        readers.pop_front();
                    }
                    let mut reader = TiffReader::open(&paths[index])?;
                    reader.set_cache_budget(8 * 1024 * 1024);
                    readers.push_back((index, reader));
                }
                let reader = &mut readers.back_mut().unwrap().1;
                // Coordinates use whole lattice indices, exactly as paint_mosaic.
                let cs: Vec<f64> = (0..w)
                    .map(|c| {
                        let x = ox + ((col0 + (out_col + c) as i64) as f64 + 0.5) * rx;
                        ((x - t[2]) / t[0]).floor()
                    })
                    .collect();
                let rs: Vec<f64> = (0..h)
                    .map(|r| {
                        let y = oy - ((row0 + (out_row + r) as i64) as f64 + 0.5) * ry;
                        ((y - t[5]) / t[4]).floor()
                    })
                    .collect();
                let cb: Vec<usize> = cs
                    .iter()
                    .filter(|c| **c >= 0.0 && **c < reader.width as f64)
                    .map(|c| *c as usize)
                    .collect();
                let rb: Vec<usize> = rs
                    .iter()
                    .filter(|r| **r >= 0.0 && **r < reader.height as f64)
                    .map(|r| *r as usize)
                    .collect();
                if cb.is_empty() || rb.is_empty() {
                    continue;
                }
                let c0 = *cb.iter().min().unwrap();
                let c1 = *cb.iter().max().unwrap();
                let r0 = *rb.iter().min().unwrap();
                let r1 = *rb.iter().max().unwrap();
                let sw = c1 - c0 + 1;
                let source = reader.read_window_raw(c0, r0, sw, r1 - r0 + 1)?;
                for (r, sr) in rs.iter().enumerate() {
                    if *sr < 0.0 || *sr >= reader.height as f64 {
                        continue;
                    }
                    for (c, sc) in cs.iter().enumerate() {
                        if *sc < 0.0 || *sc >= reader.width as f64 || !out[r * w + c].is_nan() {
                            continue;
                        }
                        let value = source[(*sr as usize - r0) * sw + *sc as usize - c0];
                        if value.is_nan() || reader.nodata == Some(value) {
                            continue;
                        }
                        out[r * w + c] = value;
                    }
                }
            }
            for value in &mut out {
                if source_nodata == Some(*value) {
                    *value = f64::NAN;
                }
                if value.is_nan() {
                    holes += 1;
                    if let Some(fill) = fill {
                        *value = fill as f32 as f64;
                    }
                } else if resolution.is_some() {
                    *value = *value as f32 as f64;
                }
            }
            Ok(out)
        },
    )?;
    Ok((ny, nx, holes, first_nodata))
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
fn containing_index(source_ny: usize, source_nx: usize, col_f: f64, row_f: f64) -> Option<usize> {
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
    if !c0.is_finite() || !r0.is_finite() || !c1.is_finite() || !r1.is_finite() {
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
    let (cols, rows) = corner_grid(source, dst_crs, &dst_transform, dst_ny, dst_nx)?;
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
                        let reached = cell_box(&cols, &rows, width, row, col).and_then(
                            |(min_c, max_c, min_r, max_r)| {
                                let (c0, c1) = gdal_span(min_c, max_c, source.nx)?;
                                let (r0, r1) = gdal_span(min_r, max_r, source.ny)?;
                                let mut sum = 0.0f64;
                                let mut count = 0u64;
                                for j in r0..=r1 {
                                    for i in c0..=c1 {
                                        let value = source.values[j * source.nx + i];
                                        if !value.is_nan() {
                                            sum += value;
                                            count += 1;
                                        }
                                    }
                                }
                                (count > 0).then(|| sum / count as f64)
                            },
                        );
                        *slot = reached
                            .or_else(|| bilinear_sample(source, centre_c, centre_r))
                            .or_else(|| {
                                containing_index(source.ny, source.nx, centre_c, centre_r)
                                    .map(|at| source.values[at])
                                    .filter(|value| !value.is_nan())
                            })
                            .unwrap_or(f64::NAN);
                    }
                    Resampling::Bilinear => {
                        *slot = bilinear_sample(source, centre_c, centre_r).unwrap_or(f64::NAN);
                    }
                    Resampling::Nearest => {
                        *slot = containing_index(source.ny, source.nx, centre_c, centre_r)
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
    Ok(Grid2 {
        ny: dst_ny,
        nx: dst_nx,
        data,
    })
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
    let (cols, rows) = corner_grid(source, dst_crs, &dst_transform, dst_ny, dst_nx)?;
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
                if let Some((min_c, max_c, min_r, max_r)) = cell_box(&cols, &rows, width, row, col)
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
                        out[category * dst_nx + col] = counts[category] as f64 / total as f64;
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
                if let Some(at) = containing_index(source.ny, source.nx, centre_c, centre_r) {
                    if valid[at] {
                        for category in 0..category_count {
                            out[category * dst_nx + col] = if category as i16 + 1 == values[at] {
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
    Ok(Stack3 {
        planes: category_count,
        ny: dst_ny,
        nx: dst_nx,
        data,
    })
}

// Source-window warps keep destination corners in the original global pixel
// coordinates. Re-basing affine transforms would change boundary rounding and
// can change an averaged cell's bytes.
const WARP_BLOCK: usize = 32;
const MAX_WINDOW_PIXELS: usize = 4 * 1024 * 1024;
pub const CONTINUOUS_WORKER_CAP: usize = 4;
pub const CONTINUOUS_READER_CACHE_BYTES: usize = 8 * 1024 * 1024;

/// Respect a smaller CPU guard while never taking the case's 48-worker budget.
pub fn continuous_worker_limit() -> usize {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|count| *count > 0)
        .unwrap_or(CONTINUOUS_WORKER_CAP)
        .min(CONTINUOUS_WORKER_CAP)
}

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_static_continuous_warp_worker_cap() -> u32 {
    CONTINUOUS_WORKER_CAP as u32
}
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_static_continuous_warp_workers() -> u32 {
    continuous_worker_limit() as u32
}
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_static_continuous_warp_source_window_bytes() -> u64 {
    (MAX_WINDOW_PIXELS * std::mem::size_of::<f64>()) as u64
}
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_static_continuous_warp_reader_cache_bytes() -> u64 {
    CONTINUOUS_READER_CACHE_BYTES as u64
}

pub struct WarpBlock {
    pub row: usize,
    pub col: usize,
    pub ny: usize,
    pub nx: usize,
    pub source_row: usize,
    pub source_col: usize,
    pub source_ny: usize,
    pub source_nx: usize,
    cols: Vec<f64>,
    rows: Vec<f64>,
}

fn for_each_warp_block(
    source: &Raster,
    dst_crs: &Crs,
    dst_transform: [f64; 6],
    dst_ny: usize,
    dst_nx: usize,
    mut consume: impl FnMut(&WarpBlock) -> Result<()>,
) -> Result<()> {
    for_each_warp_block_rows(source, dst_crs, dst_transform, dst_nx, 0, dst_ny, consume)
}

#[allow(clippy::too_many_arguments)]
fn for_each_warp_block_rows(
    source: &Raster,
    dst_crs: &Crs,
    dst_transform: [f64; 6],
    dst_nx: usize,
    row_start: usize,
    row_end: usize,
    mut consume: impl FnMut(&WarpBlock) -> Result<()>,
) -> Result<()> {
    let inv_dst = dst_crs.point_projection()?;
    let fwd_src = source.crs.point_projection()?;
    for row0 in (row_start..row_end).step_by(WARP_BLOCK) {
        for col0 in (0..dst_nx).step_by(WARP_BLOCK) {
            let mut pending = vec![(
                row0,
                col0,
                WARP_BLOCK.min(row_end - row0),
                WARP_BLOCK.min(dst_nx - col0),
            )];
            while let Some((row, col, ny, nx)) = pending.pop() {
                let mut cols = Vec::with_capacity((ny + 1) * (nx + 1));
                let mut rows = Vec::with_capacity((ny + 1) * (nx + 1));
                for j in row..=row + ny {
                    let y = dst_transform[5] + dst_transform[4] * j as f64;
                    for i in col..=col + nx {
                        let x = dst_transform[2] + dst_transform[0] * i as f64;
                        let (lon, lat) = inv_dst.inverse(x, y);
                        let (sx, sy) = fwd_src.forward(lon, lat);
                        cols.push((sx - source.transform[2]) / source.transform[0]);
                        rows.push((sy - source.transform[5]) / source.transform[4]);
                    }
                }
                let limits = |values: &[f64], len: usize| {
                    let lo = values
                        .iter()
                        .copied()
                        .filter(|v| v.is_finite())
                        .fold(f64::INFINITY, f64::min);
                    let hi = values
                        .iter()
                        .copied()
                        .filter(|v| v.is_finite())
                        .fold(f64::NEG_INFINITY, f64::max);
                    let first = (lo.floor() - 2.0).max(0.0).min(len as f64) as usize;
                    let end = (hi.ceil() + 2.0).max(0.0).min(len as f64) as usize;
                    (first, end.saturating_sub(first))
                };
                let (source_col, source_nx) = limits(&cols, source.nx);
                let (source_row, source_ny) = limits(&rows, source.ny);
                if source_nx.saturating_mul(source_ny) > MAX_WINDOW_PIXELS {
                    if nx == 1 && ny == 1 {
                        return Err(StaticError::Invalid(
                            "one destination cell exceeds the 32 MiB source-window budget".into(),
                        ));
                    }
                    if nx >= ny && nx > 1 {
                        let first = nx / 2;
                        pending.push((row, col + first, ny, nx - first));
                        pending.push((row, col, ny, first));
                    } else {
                        let first = ny / 2;
                        pending.push((row + first, col, ny - first, nx));
                        pending.push((row, col, first, nx));
                    }
                    continue;
                }
                consume(&WarpBlock {
                    row,
                    col,
                    ny,
                    nx,
                    source_row,
                    source_col,
                    source_ny,
                    source_nx,
                    cols,
                    rows,
                })?;
            }
        }
    }
    Ok(())
}

fn window_index(block: &WarpBlock, row: usize, col: usize) -> usize {
    (row - block.source_row) * block.source_nx + col - block.source_col
}

/// Byte-identical continuous warp using a bounded source window per block.
#[allow(clippy::too_many_arguments)]
pub fn reproject_continuous_windowed(
    source: &Raster,
    dst_crs: &Crs,
    dst_transform: [f64; 6],
    dst_ny: usize,
    dst_nx: usize,
    method: Resampling,
    mut read: impl FnMut(usize, usize, usize, usize) -> Result<Vec<f64>>,
) -> Result<Grid2> {
    let mut data = vec![f64::NAN; dst_ny * dst_nx];
    for_each_warp_block(source, dst_crs, dst_transform, dst_ny, dst_nx, |b| {
        let values = read(b.source_col, b.source_row, b.source_nx, b.source_ny)?;
        write_continuous_block(source, b, &values, method, dst_nx, dst_ny, &mut data);
        Ok(())
    })?;
    Ok(Grid2 {
        ny: dst_ny,
        nx: dst_nx,
        data,
    })
}

/// One worker writes a disjoint south-to-north output band. Corners and
/// source scan order use global destination indices, without an affine shift.
#[allow(clippy::too_many_arguments)]
fn write_continuous_block(
    source: &Raster,
    b: &WarpBlock,
    values: &[f64],
    method: Resampling,
    dst_nx: usize,
    row_end: usize,
    data: &mut [f64],
) {
    let at = |j, i| values[window_index(b, j, i)];
    let containing = |c, r| {
        containing_index(source.ny, source.nx, c, r)
            .map(|index| at(index / source.nx, index % source.nx))
    };
    let bilinear = |c: f64, r: f64| -> Option<f64> {
        let px = c - 0.5;
        let py = r - 0.5;
        let i = px.floor();
        let j = py.floor();
        if i < 0.0
            || j < 0.0
            || i + 1.0 > source.nx as f64 - 1.0
            || j + 1.0 > source.ny as f64 - 1.0
        {
            return None;
        }
        if !i.is_finite() || !j.is_finite() {
            return None;
        }
        let (i, j) = (i as usize, j as usize);
        let fx = px - i as f64;
        let fy = py - j as f64;
        let (v00, v01, v10, v11) = (at(j, i), at(j, i + 1), at(j + 1, i), at(j + 1, i + 1));
        if v00.is_nan() || v01.is_nan() || v10.is_nan() || v11.is_nan() {
            return None;
        }
        Some(
            v00 * (1.0 - fx) * (1.0 - fy)
                + v01 * fx * (1.0 - fy)
                + v10 * (1.0 - fx) * fy
                + v11 * fx * fy,
        )
    };
    let width = b.nx + 1;
    for row in 0..b.ny {
        for col in 0..b.nx {
            let cc = 0.25
                * (b.cols[row * width + col]
                    + b.cols[row * width + col + 1]
                    + b.cols[(row + 1) * width + col]
                    + b.cols[(row + 1) * width + col + 1]);
            let cr = 0.25
                * (b.rows[row * width + col]
                    + b.rows[row * width + col + 1]
                    + b.rows[(row + 1) * width + col]
                    + b.rows[(row + 1) * width + col + 1]);
            let value = match method {
                Resampling::Average => cell_box(&b.cols, &b.rows, width, row, col)
                    .and_then(|(c0, c1, r0, r1)| {
                        let (c0, c1) = gdal_span(c0, c1, source.nx)?;
                        let (r0, r1) = gdal_span(r0, r1, source.ny)?;
                        let mut sum = 0.0f64;
                        let mut count = 0u64;
                        for j in r0..=r1 {
                            for i in c0..=c1 {
                                let v = at(j, i);
                                if !v.is_nan() {
                                    sum += v;
                                    count += 1;
                                }
                            }
                        }
                        (count > 0).then(|| sum / count as f64)
                    })
                    .or_else(|| bilinear(cc, cr))
                    .or_else(|| containing(cc, cr).filter(|v| !v.is_nan())),
                Resampling::Bilinear => bilinear(cc, cr),
                Resampling::Nearest => containing(cc, cr),
            }
            .unwrap_or(f64::NAN);
            data[(row_end - 1 - b.row - row) * dst_nx + b.col + col] = value;
        }
    }
}

/// Fixed-worker continuous warp. Exactly one reader per output-band worker
/// is alive, with at most four source windows and four decoded-block caches.
#[allow(clippy::too_many_arguments)]
pub fn reproject_continuous_windowed_parallel<T: Send>(
    source: &Raster,
    dst_crs: &Crs,
    dst_transform: [f64; 6],
    dst_ny: usize,
    dst_nx: usize,
    method: Resampling,
    requested_workers: usize,
    make_reader: impl Fn() -> Result<T> + Sync,
    read: impl Fn(&mut T, usize, usize, usize, usize) -> Result<Vec<f64>> + Sync,
) -> Result<Grid2> {
    let mut data = vec![f64::NAN; dst_ny * dst_nx];
    if dst_ny == 0 || dst_nx == 0 {
        return Ok(Grid2 {
            ny: dst_ny,
            nx: dst_nx,
            data,
        });
    }
    let workers = requested_workers
        .clamp(1, CONTINUOUS_WORKER_CAP)
        .min(dst_ny);
    let rows_per_worker = dst_ny.div_ceil(workers);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|err| {
            StaticError::Invalid(format!(
                "cannot start bounded continuous-warp workers: {err}"
            ))
        })?;
    let errors = std::sync::Mutex::new(Vec::new());
    pool.scope(|scope| {
        for (index, output) in data.chunks_mut(rows_per_worker * dst_nx).enumerate() {
            let row_end = dst_ny - index * rows_per_worker;
            let row_start = row_end - output.len() / dst_nx;
            let (make_reader, read, errors) = (&make_reader, &read, &errors);
            scope.spawn(move |_| {
                let result = (|| {
                    let mut reader = make_reader()?;
                    for_each_warp_block_rows(
                        source,
                        dst_crs,
                        dst_transform,
                        dst_nx,
                        row_start,
                        row_end,
                        |block| {
                            let values = read(
                                &mut reader,
                                block.source_col,
                                block.source_row,
                                block.source_nx,
                                block.source_ny,
                            )?;
                            write_continuous_block(
                                source, block, &values, method, dst_nx, row_end, output,
                            );
                            Ok(())
                        },
                    )
                })();
                if let Err(error) = result {
                    errors
                        .lock()
                        .expect("continuous error lock")
                        .push((row_start, error));
                }
            });
        }
    });
    let mut errors = errors.into_inner().expect("continuous error lock");
    errors.sort_by_key(|(row, _)| *row);
    if !errors.is_empty() {
        return Err(errors.remove(0).1);
    }
    Ok(Grid2 {
        ny: dst_ny,
        nx: dst_nx,
        data,
    })
}

/// Byte-identical categorical warp without full source masks or corner planes.
#[allow(clippy::too_many_arguments)]
pub fn reproject_categories_windowed(
    source: &Raster,
    dst_crs: &Crs,
    dst_transform: [f64; 6],
    dst_ny: usize,
    dst_nx: usize,
    category_count: usize,
    mut read: impl FnMut(usize, usize, usize, usize) -> Result<Vec<i16>>,
) -> Result<Stack3> {
    let cells = dst_ny * dst_nx;
    let mut data = vec![f64::NAN; category_count * cells];
    for_each_warp_block(source, dst_crs, dst_transform, dst_ny, dst_nx, |b| {
        let values = read(b.source_col, b.source_row, b.source_nx, b.source_ny)?;
        let mut counts = vec![0u64; category_count];
        let width = b.nx + 1;
        for row in 0..b.ny {
            for col in 0..b.nx {
                counts.fill(0);
                let mut total = 0u64;
                if let Some((c0, c1, r0, r1)) = cell_box(&b.cols, &b.rows, width, row, col) {
                    if let (Some((c0, c1)), Some((r0, r1))) =
                        (gdal_span(c0, c1, source.nx), gdal_span(r0, r1, source.ny))
                    {
                        for j in r0..=r1 {
                            for i in c0..=c1 {
                                let v = values[window_index(b, j, i)];
                                if v > 0 {
                                    counts[v as usize - 1] += 1;
                                    total += 1;
                                }
                            }
                        }
                    }
                }
                if total == 0 {
                    let cc = 0.25
                        * (b.cols[row * width + col]
                            + b.cols[row * width + col + 1]
                            + b.cols[(row + 1) * width + col]
                            + b.cols[(row + 1) * width + col + 1]);
                    let cr = 0.25
                        * (b.rows[row * width + col]
                            + b.rows[row * width + col + 1]
                            + b.rows[(row + 1) * width + col]
                            + b.rows[(row + 1) * width + col + 1]);
                    if let Some(at) = containing_index(source.ny, source.nx, cc, cr) {
                        let v = values[window_index(b, at / source.nx, at % source.nx)];
                        if v > 0 {
                            counts[v as usize - 1] = 1;
                            total = 1;
                        }
                    }
                }
                if total > 0 {
                    for category in 0..category_count {
                        data[category * cells
                            + (dst_ny - 1 - b.row - row) * dst_nx
                            + b.col
                            + col] = counts[category] as f64 / total as f64;
                    }
                }
            }
        }
        Ok(())
    })?;
    Ok(Stack3 {
        planes: category_count,
        ny: dst_ny,
        nx: dst_nx,
        data,
    })
}

#[cfg(test)]
mod mosaic_lattice_regression {
    use super::*;

    /// A 40x40 point-sampled source: pixel centres on whole multiples of
    /// 0.125 degrees, value = its own index.
    fn source() -> Raster {
        Raster {
            ny: 40,
            nx: 40,
            values: (0..1600).map(|i| i as f64).collect(),
            transform: [0.125, 0.0, -1.0625, 0.0, -0.125, 2.0625],
            crs: Crs::Geographic,
        }
    }

    /// Two footprints whose edges sit at different sub-pixel offsets.
    const FOOTPRINTS: [[f64; 4]; 2] = [[-0.93, -1.71, 1.37, 1.83], [-0.61, -1.52, 2.19, 1.64]];

    fn assert_shared_ground_agrees(a: &Raster, b: &Raster, res: f64) {
        let dx = (b.transform[2] - a.transform[2]) / res;
        let dy = (a.transform[5] - b.transform[5]) / res;
        assert_eq!(dx, dx.round(), "column offset is whole pixels");
        assert_eq!(dy, dy.round(), "row offset is whole pixels");
        let (dx, dy) = (dx.round() as usize, dy.round() as usize);
        let mut shared = 0usize;
        for j in 0..b.ny.min(a.ny - dy) {
            for i in 0..b.nx.min(a.nx - dx) {
                let (va, vb) = (a.values[(j + dy) * a.nx + i + dx], b.values[j * b.nx + i]);
                assert_eq!(va.to_bits(), vb.to_bits(), "value at {i},{j}");
                let (ca, cb) = (a.centre(i + dx, j + dy), b.centre(i, j));
                assert_eq!(ca.0.to_bits(), cb.0.to_bits(), "x at {i},{j}");
                assert_eq!(ca.1.to_bits(), cb.1.to_bits(), "y at {i},{j}");
                shared += 1;
            }
        }
        assert!(shared > 100, "the footprints must overlap ({shared})");
    }

    #[test]
    fn declared_resolution_crops_agree_on_shared_ground() {
        let r = 0.125;
        let windows: Vec<Raster> = FOOTPRINTS
            .iter()
            .map(|bounds| mosaic(&[source()], *bounds, Some(r), None).unwrap().0)
            .collect();
        assert_shared_ground_agrees(&windows[0], &windows[1], r);
        // Every output centre IS a source sample: nothing is a tie.
        let src = source();
        let w = &windows[0];
        for j in 0..w.ny {
            for i in 0..w.nx {
                let (x, y) = w.centre(i, j);
                let col = ((x + 1.0) / r).round() as usize;
                let row = ((2.0 - y) / r).round() as usize;
                assert_eq!(w.values[j * w.nx + i], src.values[row * 40 + col]);
            }
        }
    }

    #[test]
    fn inherited_resolution_crops_keep_the_source_lattice() {
        let src = source();
        let t = src.transform;
        let windows: Vec<Raster> = FOOTPRINTS
            .iter()
            .map(|bounds| mosaic(&[src.clone()], *bounds, None, None).unwrap().0)
            .collect();
        assert_shared_ground_agrees(&windows[0], &windows[1], t[0]);
        for w in &windows {
            let col = (w.transform[2] - t[2]) / t[0];
            let row = (t[5] - w.transform[5]) / t[0];
            assert_eq!(col, col.round());
            assert_eq!(row, row.round());
        }
    }

    #[test]
    fn a_crop_covers_its_whole_footprint() {
        let r = 0.125;
        for bounds in FOOTPRINTS {
            let (w, _) = mosaic(&[source()], bounds, Some(r), None).unwrap();
            let [west, south, east, north] = w.bounds();
            assert!(west <= bounds[0] && south <= bounds[1]);
            assert!(east >= bounds[2] && north >= bounds[3]);
            assert!(bounds[0] - west < r && north - bounds[3] < r);
        }
    }
}
