//! Run-level grid file (`.rwg`) and the generic lat/lon -> fractional grid
//! coordinate locator that makes point products (soundings, meteograms) work
//! at arbitrary click points on any smooth, invertible model grid.
//!
//! `.rwg` byte layout (little-endian, 64-byte header):
//! ```text
//!  0- 7  magic         b"RWSGRID1"
//!  8-11  version       u32
//! 12-15  meta_len      u32
//! 16-23  lat_comp_len  u64
//! 24-31  lon_comp_len  u64
//! 32-63  reserved      (zeros)
//! ```
//! followed by `[meta JSON][zstd-1 of lat f32 LE bytes][zstd-1 of lon f32 LE
//! bytes]`. The grid hash (sha256 hex of the full file bytes) is the identity
//! that hour files and run manifests reference via `grid_hash`.

use std::fmt::Write as _;
use std::fs::File;
use std::io::Read;
use std::mem::size_of;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use rustwx_core::{GridProjection, LatLonGrid, MAX_GRID_CELLS};

use crate::atomic::atomic_write_bytes;
use crate::error::{RwResult, RwStoreError};
use crate::format::HEADER_LEN;

/// Magic bytes at the start of every `.rwg` grid file.
pub const GRID_MAGIC: &[u8; 8] = b"RWSGRID1";
/// Current grid-file format version.
pub const GRID_VERSION: u32 = 1;
/// Grid-file format versions this build can read.
pub const GRID_SUPPORTED_VERSIONS: &[u32] = &[1];
/// Schema identifier embedded in grid-file metadata.
pub const SCHEMA_GRID: &str = "rw-store.grid.v1";

/// zstd compression level for coordinate arrays (cheap, they compress well).
const ZSTD_LEVEL: i32 = 1;
/// Maximum JSON metadata block accepted from a grid file. Projection metadata
/// is normally only a few hundred bytes; one MiB leaves ample forward room
/// without allowing a sparse file to make serde scan gigabytes.
const MAX_GRID_META_LEN: u64 = 1024 * 1024;
/// One coordinate per supported horizontal grid cell.
const MAX_COORD_RAW_LEN: u64 = MAX_GRID_CELLS as u64 * size_of::<f32>() as u64;
/// zstd's incompressible-frame overhead is far below one MiB at this size.
/// Keeping the allowance explicit rejects padded/skippable-frame bombs while
/// accepting every file emitted by this crate at the grid ceiling.
const MAX_COORD_COMP_LEN: u64 = MAX_COORD_RAW_LEN + 1024 * 1024;
/// 128 MiB is above the largest supported raw coordinate array and bounds the
/// zstd decoder's history allocation for hostile frame headers.
const MAX_GRID_WINDOW_LOG: u32 = 27;
/// Absolute `.rwg` size ceiling derived from the bounded sections above.
const MAX_GRID_FILE_LEN: u64 = HEADER_LEN as u64 + MAX_GRID_META_LEN + 2 * MAX_COORD_COMP_LEN;

#[derive(Debug, Clone, Copy)]
struct GridLayout {
    meta_end: usize,
    lat_end: usize,
    file_end: usize,
}

fn try_zeroed_bytes(len: usize, what: &str) -> RwResult<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).map_err(|err| {
        RwStoreError::Format(format!("cannot allocate {len} bytes for {what}: {err}"))
    })?;
    bytes.resize(len, 0);
    Ok(bytes)
}

/// Validate every header-declared section against both the actual file and
/// the format ceilings before allocating a file-sized buffer.
fn parse_grid_layout(header: &[u8; HEADER_LEN], file_len: u64) -> RwResult<GridLayout> {
    if &header[0..8] != GRID_MAGIC.as_slice() {
        return Err(RwStoreError::Format(format!(
            "bad grid magic: expected {GRID_MAGIC:?}, got {:?}",
            &header[0..8]
        )));
    }
    let version = u32::from_le_bytes(header[8..12].try_into().unwrap());
    if !GRID_SUPPORTED_VERSIONS.contains(&version) {
        return Err(RwStoreError::UnsupportedVersion {
            found: version,
            supported: GRID_SUPPORTED_VERSIONS,
        });
    }

    let meta_len = u64::from(u32::from_le_bytes(header[12..16].try_into().unwrap()));
    let lat_comp_len = u64::from_le_bytes(header[16..24].try_into().unwrap());
    let lon_comp_len = u64::from_le_bytes(header[24..32].try_into().unwrap());
    if meta_len > MAX_GRID_META_LEN {
        return Err(RwStoreError::Format(format!(
            "grid meta JSON is {meta_len} bytes; limit is {MAX_GRID_META_LEN} bytes"
        )));
    }
    for (name, len) in [("lat", lat_comp_len), ("lon", lon_comp_len)] {
        if len > MAX_COORD_COMP_LEN {
            return Err(RwStoreError::Format(format!(
                "grid {name} compressed section is {len} bytes; limit is {MAX_COORD_COMP_LEN} bytes"
            )));
        }
    }

    let meta_end = (HEADER_LEN as u64).checked_add(meta_len);
    let lat_end = meta_end.and_then(|end| end.checked_add(lat_comp_len));
    let file_end = lat_end.and_then(|end| end.checked_add(lon_comp_len));
    let (meta_end, lat_end, file_end) = match (meta_end, lat_end, file_end) {
        (Some(meta_end), Some(lat_end), Some(file_end)) => (meta_end, lat_end, file_end),
        _ => {
            return Err(RwStoreError::Format(
                "grid section lengths overflow the file layout".to_string(),
            ));
        }
    };
    if file_end != file_len {
        return Err(RwStoreError::Format(format!(
            "grid file length mismatch: header describes {file_end} bytes, have {file_len}"
        )));
    }

    Ok(GridLayout {
        meta_end: usize::try_from(meta_end).map_err(|_| {
            RwStoreError::Format(format!("grid meta end {meta_end} does not fit usize"))
        })?,
        lat_end: usize::try_from(lat_end).map_err(|_| {
            RwStoreError::Format(format!("grid latitude end {lat_end} does not fit usize"))
        })?,
        file_end: usize::try_from(file_end).map_err(|_| {
            RwStoreError::Format(format!("grid file length {file_end} does not fit usize"))
        })?,
    })
}

fn validate_grid_file_len(file_len: u64) -> RwResult<()> {
    if file_len < HEADER_LEN as u64 {
        return Err(RwStoreError::Format(format!(
            "grid header requires {HEADER_LEN} bytes, got {file_len}"
        )));
    }
    if file_len > MAX_GRID_FILE_LEN {
        return Err(RwStoreError::Format(format!(
            "grid file is {file_len} bytes; limit is {MAX_GRID_FILE_LEN} bytes"
        )));
    }
    Ok(())
}

fn read_grid_bytes(path: &Path) -> RwResult<(Vec<u8>, GridLayout)> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    validate_grid_file_len(file_len)?;

    let mut header = [0u8; HEADER_LEN];
    file.read_exact(&mut header)?;
    let layout = parse_grid_layout(&header, file_len)?;
    let mut data = try_zeroed_bytes(layout.file_end, "grid file")?;
    data[..HEADER_LEN].copy_from_slice(&header);
    file.read_exact(&mut data[HEADER_LEN..])?;

    // Detect growth between metadata() and the read instead of silently
    // hashing/decoding only the old prefix.
    let mut trailing = [0u8; 1];
    if file.read(&mut trailing)? != 0 {
        return Err(RwStoreError::Format(
            "grid file grew while it was being read".to_string(),
        ));
    }
    Ok((data, layout))
}

/// Grid-file metadata stored as JSON after the header.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RwsGridMeta {
    pub schema: String,
    pub nx: usize,
    pub ny: usize,
    pub lat_raw_len: u64,
    pub lon_raw_len: u64,
    pub projection: Option<GridProjection>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

fn f32s_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Atomically write `grid` (+ optional projection) as a `.rwg` file and
/// return the sha256 hex digest of the final file bytes.
pub fn write_grid(
    path: &Path,
    grid: &LatLonGrid,
    projection: Option<&GridProjection>,
) -> RwResult<String> {
    let (bytes, hash) = encode_grid_bytes(grid, projection)?;
    atomic_write_bytes(path, &bytes)?;
    Ok(hash)
}

/// Build the exact `.rwg` byte image [`write_grid`] would write, plus its
/// sha256 hex hash, without touching disk — for writers that need the grid
/// hash up front but want to defer the file write to their commit point.
pub fn encode_grid_bytes(
    grid: &LatLonGrid,
    projection: Option<&GridProjection>,
) -> RwResult<(Vec<u8>, String)> {
    let nx = grid.shape.nx;
    let ny = grid.shape.ny;
    if nx == 0 || ny == 0 {
        return Err(RwStoreError::Grid(format!(
            "degenerate grid {nx}x{ny} (nx and ny must be nonzero)"
        )));
    }
    let cells = nx
        .checked_mul(ny)
        .ok_or_else(|| RwStoreError::Grid(format!("grid {nx}x{ny} overflows the cell count")))?;
    if cells > MAX_GRID_CELLS {
        return Err(RwStoreError::Grid(format!(
            "grid {nx}x{ny} has {cells} cells; limit is {MAX_GRID_CELLS}"
        )));
    }
    if grid.lat_deg.len() != cells || grid.lon_deg.len() != cells {
        return Err(RwStoreError::Grid(format!(
            "coordinate arrays must hold {cells} values ({ny} x {nx}), \
             got lat {} / lon {}",
            grid.lat_deg.len(),
            grid.lon_deg.len()
        )));
    }

    let lat_raw = f32s_to_le_bytes(&grid.lat_deg);
    let lon_raw = f32s_to_le_bytes(&grid.lon_deg);
    let lat_comp = zstd::stream::encode_all(&lat_raw[..], ZSTD_LEVEL)?;
    let lon_comp = zstd::stream::encode_all(&lon_raw[..], ZSTD_LEVEL)?;

    let meta = RwsGridMeta {
        schema: SCHEMA_GRID.to_string(),
        nx,
        ny,
        lat_raw_len: lat_raw.len() as u64,
        lon_raw_len: lon_raw.len() as u64,
        projection: projection.cloned(),
    };
    let meta_bytes =
        serde_json::to_vec(&meta).map_err(|err| RwStoreError::Meta(err.to_string()))?;
    let meta_len = u32::try_from(meta_bytes.len()).map_err(|_| {
        RwStoreError::Format(format!(
            "grid meta JSON too large: {} bytes",
            meta_bytes.len()
        ))
    })?;

    let mut header = [0u8; HEADER_LEN];
    header[0..8].copy_from_slice(GRID_MAGIC); // 0-7
    header[8..12].copy_from_slice(&GRID_VERSION.to_le_bytes()); // 8-11
    header[12..16].copy_from_slice(&meta_len.to_le_bytes()); // 12-15
    header[16..24].copy_from_slice(&(lat_comp.len() as u64).to_le_bytes()); // 16-23
    header[24..32].copy_from_slice(&(lon_comp.len() as u64).to_le_bytes()); // 24-31
    // 32-63 already zeroed

    let mut bytes =
        Vec::with_capacity(HEADER_LEN + meta_bytes.len() + lat_comp.len() + lon_comp.len());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&meta_bytes);
    bytes.extend_from_slice(&lat_comp);
    bytes.extend_from_slice(&lon_comp);

    let hash = sha256_hex(&bytes);
    Ok((bytes, hash))
}

/// Decoded contents of a `.rwg` grid file. `lat`/`lon` are row-major
/// (`ny` rows of `nx`) and `hash` is sha256 hex of the file bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct GridFile {
    pub nx: usize,
    pub ny: usize,
    pub lat: Vec<f32>,
    pub lon: Vec<f32>,
    pub projection: Option<GridProjection>,
    pub hash: String,
}

impl GridFile {
    /// Whether stored row 0 is the NORTHERNMOST row, i.e. latitude decreases
    /// as the row index grows. Derived from the data, never assumed: scan
    /// down each column for the first pair of distinct finite latitudes
    /// (NaN-safe — masked grids can have whole NaN rows/corners). `None`
    /// when nothing decides (all-NaN or constant latitude).
    ///
    /// Display code should flip rows only when this is `Some(false)` /
    /// defaulted-false: south-to-north storage needs a flip to render
    /// north-up; north-to-south storage is already north-up.
    pub fn lat_descending(&self) -> Option<bool> {
        for x in 0..self.nx {
            let mut first: Option<f32> = None;
            for y in 0..self.ny {
                let lat = self.lat[y * self.nx + x];
                if !lat.is_finite() {
                    continue;
                }
                match first {
                    None => first = Some(lat),
                    Some(earlier) if lat != earlier => return Some(lat < earlier),
                    Some(_) => {}
                }
            }
        }
        None
    }

    /// Open and fully validate a `.rwg` file (magic, version, lengths, all
    /// with checked math — trust nothing on disk).
    pub fn open(path: &Path) -> RwResult<Self> {
        let (data, layout) = read_grid_bytes(path)?;
        let meta: RwsGridMeta = serde_json::from_slice(&data[HEADER_LEN..layout.meta_end])
            .map_err(|err| RwStoreError::Meta(format!("grid meta JSON: {err}")))?;
        if meta.schema != SCHEMA_GRID {
            return Err(RwStoreError::Meta(format!(
                "unexpected schema '{}' (expected '{SCHEMA_GRID}')",
                meta.schema
            )));
        }
        if meta.nx == 0 || meta.ny == 0 {
            return Err(RwStoreError::Meta(format!(
                "degenerate grid {}x{} (nx and ny must be nonzero)",
                meta.nx, meta.ny
            )));
        }
        let cells = meta
            .nx
            .checked_mul(meta.ny)
            .filter(|&cells| cells <= MAX_GRID_CELLS)
            .ok_or_else(|| {
                RwStoreError::Meta(format!(
                    "grid {}x{} exceeds the supported ceiling of {MAX_GRID_CELLS} cells",
                    meta.nx, meta.ny,
                ))
            })?;
        let raw_len = (cells as u64)
            .checked_mul(size_of::<f32>() as u64)
            .filter(|&len| len <= MAX_COORD_RAW_LEN)
            .ok_or_else(|| {
                RwStoreError::Meta(format!(
                    "grid {}x{} exceeds the supported coordinate array size",
                    meta.nx, meta.ny
                ))
            })?;
        if meta.lat_raw_len != raw_len || meta.lon_raw_len != raw_len {
            return Err(RwStoreError::Meta(format!(
                "coordinate raw lengths (lat {}, lon {}) do not match {}x{} grid ({raw_len})",
                meta.lat_raw_len, meta.lon_raw_len, meta.nx, meta.ny
            )));
        }

        let lat = decompress_coords(
            &data[layout.meta_end..layout.lat_end],
            raw_len as usize,
            "lat",
        )?;
        let lon = decompress_coords(
            &data[layout.lat_end..layout.file_end],
            raw_len as usize,
            "lon",
        )?;

        let hash = sha256_hex(&data);
        Ok(Self {
            nx: meta.nx,
            ny: meta.ny,
            lat,
            lon,
            projection: meta.projection,
            hash,
        })
    }
}

/// Decompress one coordinate section (capacity-capped at `raw_len`) and
/// decode it as little-endian f32s.
fn decompress_coords(comp: &[u8], raw_len: usize, name: &str) -> RwResult<Vec<f32>> {
    let mut raw = try_zeroed_bytes(raw_len, &format!("grid {name} coordinate bytes"))?;
    let mut decoder = zstd::stream::read::Decoder::with_buffer(comp)
        .map_err(|err| RwStoreError::Format(format!("grid {name} array decompress: {err}")))?;
    decoder
        .window_log_max(MAX_GRID_WINDOW_LOG)
        .map_err(|err| RwStoreError::Format(format!("grid {name} zstd window limit: {err}")))?;
    decoder
        .read_exact(&mut raw)
        .map_err(|err| RwStoreError::Format(format!("grid {name} array decompress: {err}")))?;
    let mut extra = [0u8; 1];
    if decoder
        .read(&mut extra)
        .map_err(|err| RwStoreError::Format(format!("grid {name} array decompress: {err}")))?
        != 0
    {
        return Err(RwStoreError::Format(format!(
            "grid {name} array decompressed beyond the expected {raw_len} bytes"
        )));
    }

    let value_count = raw_len / size_of::<f32>();
    let mut values = Vec::new();
    values.try_reserve_exact(value_count).map_err(|err| {
        RwStoreError::Format(format!(
            "cannot allocate {value_count} f32 values for grid {name} coordinates: {err}"
        ))
    })?;
    for chunk in raw.chunks_exact(size_of::<f32>()) {
        values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(values)
}

/// Coarse-mesh sampling stride for the locator (every 8th grid point).
const LOCATOR_STRIDE: usize = 8;
/// Fine-scan half-window around the winning coarse sample, in grid points.
const REFINE_RADIUS: usize = 8;
/// Reject queries whose best coarse distance exceeds this many local
/// coarse-cell diagonals (in weighted degrees).
const MAX_COARSE_DIAGONALS: f64 = 4.0;
/// Floor for the cos(lat) longitude weight so polar queries stay finite.
const MIN_COS_LAT: f64 = 0.05;
/// Newton iterations for the inverse-bilinear sub-cell solve.
const NEWTON_ITERATIONS: usize = 3;
/// Accept Newton solutions within this tolerance of the unit cell.
const CELL_TOLERANCE: f64 = 1e-3;

/// Squared "weighted degree" distance: dlon is wrapped to [-180, 180) and
/// scaled by cos(query lat) so high-latitude lon spans do not dominate.
fn dist2(lat_a: f64, lon_a: f64, lat_b: f64, lon_b: f64, lon_weight: f64) -> f64 {
    let dlat = lat_a - lat_b;
    let dlon = wrap_dlon(lon_a - lon_b) * lon_weight;
    dlat * dlat + dlon * dlon
}

/// Wrap a longitude difference into [-180, 180).
fn wrap_dlon(dlon: f64) -> f64 {
    (dlon + 180.0).rem_euclid(360.0) - 180.0
}

/// Generic lat/lon -> fractional grid coordinate inverter. Works on any
/// smooth, locally invertible curvilinear grid (regular lat/lon, Lambert,
/// rotated, ...) without knowing the projection: coarse nearest-sample scan,
/// fine window refinement, then inverse-bilinear Newton inside one cell.
pub struct GridLocator {
    nx: usize,
    ny: usize,
    lat: Vec<f32>,
    lon: Vec<f32>,
    /// Sampled x indices (every `LOCATOR_STRIDE`, always including nx-1).
    sample_xs: Vec<usize>,
    /// Sampled y indices (every `LOCATOR_STRIDE`, always including ny-1).
    sample_ys: Vec<usize>,
    /// Coarse-mesh latitudes, row-major over (sample_ys, sample_xs).
    coarse_lat: Vec<f32>,
    /// Coarse-mesh longitudes, row-major over (sample_ys, sample_xs).
    coarse_lon: Vec<f32>,
}

fn sample_indices(n: usize, stride: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).step_by(stride).collect();
    if *indices.last().expect("n >= 1") != n - 1 {
        indices.push(n - 1);
    }
    indices
}

impl GridLocator {
    /// Build the coarse search mesh by sampling every `LOCATOR_STRIDE`-th
    /// point (last row/column always included).
    pub fn build(grid: &GridFile) -> Self {
        let sample_xs = sample_indices(grid.nx, LOCATOR_STRIDE);
        let sample_ys = sample_indices(grid.ny, LOCATOR_STRIDE);
        let mut coarse_lat = Vec::with_capacity(sample_xs.len() * sample_ys.len());
        let mut coarse_lon = Vec::with_capacity(sample_xs.len() * sample_ys.len());
        for &y in &sample_ys {
            for &x in &sample_xs {
                coarse_lat.push(grid.lat[y * grid.nx + x]);
                coarse_lon.push(grid.lon[y * grid.nx + x]);
            }
        }
        Self {
            nx: grid.nx,
            ny: grid.ny,
            lat: grid.lat.clone(),
            lon: grid.lon.clone(),
            sample_xs,
            sample_ys,
            coarse_lat,
            coarse_lon,
        }
    }

    /// Invert (lat, lon) to fractional grid coordinates (fx, fy), where
    /// integer values land exactly on grid points. Returns `None` for
    /// non-finite input or points far outside the grid. Allocation-free.
    ///
    /// Known limit: on lon-periodic 0-360 grids (e.g. GFS global) the
    /// one-cell-wide seam between the last and first column has no
    /// bracketing cell, so queries there fall back to the nearest grid
    /// point (worst case ~half a cell off). Dateline-CROSSING cells invert
    /// fine; only the periodic wrap cell is unmodeled. Revisit if sub-cell
    /// seam accuracy ever matters.
    pub fn locate(&self, lat: f64, lon: f64) -> Option<(f64, f64)> {
        if !lat.is_finite() || !lon.is_finite() {
            return None;
        }
        let lon_weight = lat.to_radians().cos().abs().max(MIN_COS_LAT);

        // (a) Coarse pass: nearest finite sample on the downsampled mesh.
        let mut best: Option<(f64, usize, usize)> = None;
        for sj in 0..self.sample_ys.len() {
            for si in 0..self.sample_xs.len() {
                let k = sj * self.sample_xs.len() + si;
                let slat = self.coarse_lat[k] as f64;
                let slon = self.coarse_lon[k] as f64;
                if !slat.is_finite() || !slon.is_finite() {
                    continue;
                }
                let d2 = dist2(lat, lon, slat, slon, lon_weight);
                if best.is_none_or(|(best_d2, _, _)| d2 < best_d2) {
                    best = Some((d2, si, sj));
                }
            }
        }
        let (best_d2, si, sj) = best?;

        // Sanity gate: reject queries farther than a few coarse-cell
        // diagonals from every sample — they are outside the grid.
        let diag2 = self.coarse_diag2(si, sj, lon_weight);
        let threshold2 = if diag2 > 0.0 {
            MAX_COARSE_DIAGONALS * MAX_COARSE_DIAGONALS * diag2
        } else {
            1.0 // degenerate mesh (single sample / NaN neighbors): 1 deg^2
        };
        if best_d2 > threshold2 {
            return None;
        }

        // (b) Refine: exhaustive scan of the fine window around the sample.
        let center_x = self.sample_xs[si];
        let center_y = self.sample_ys[sj];
        let x0 = center_x.saturating_sub(REFINE_RADIUS);
        let x1 = (center_x + REFINE_RADIUS).min(self.nx - 1);
        let y0 = center_y.saturating_sub(REFINE_RADIUS);
        let y1 = (center_y + REFINE_RADIUS).min(self.ny - 1);
        let mut fine_best: Option<(f64, usize, usize)> = None;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let idx = y * self.nx + x;
                let slat = self.lat[idx] as f64;
                let slon = self.lon[idx] as f64;
                if !slat.is_finite() || !slon.is_finite() {
                    continue;
                }
                let d2 = dist2(lat, lon, slat, slon, lon_weight);
                if fine_best.is_none_or(|(best_d2, _, _)| d2 < best_d2) {
                    fine_best = Some((d2, x, y));
                }
            }
        }
        let (_, px, py) = fine_best?;

        // (c) Sub-cell: try the up-to-4 cells touching the winning fine
        // point; the one whose corners bracket the query yields an in-range
        // inverse-bilinear solution.
        if self.nx >= 2 && self.ny >= 2 {
            for &cell_y in &[py.wrapping_sub(1), py] {
                if cell_y >= self.ny - 1 {
                    continue;
                }
                for &cell_x in &[px.wrapping_sub(1), px] {
                    if cell_x >= self.nx - 1 {
                        continue;
                    }
                    if let Some((s, t)) = self.invert_cell(cell_x, cell_y, lat, lon) {
                        return Some((cell_x as f64 + s, cell_y as f64 + t));
                    }
                }
            }
        }
        // Fallback: nearest fine point with zero fractional part.
        Some((px as f64, py as f64))
    }

    /// Squared weighted-degree diagonal of the coarse cell at sample
    /// (si, sj), from the largest finite-neighbor spacing per axis.
    fn coarse_diag2(&self, si: usize, sj: usize, lon_weight: f64) -> f64 {
        let stride_x = self.sample_xs.len();
        let center_k = sj * stride_x + si;
        let center_lat = self.coarse_lat[center_k] as f64;
        let center_lon = self.coarse_lon[center_k] as f64;
        let spacing2 = |k: usize| -> f64 {
            let nlat = self.coarse_lat[k] as f64;
            let nlon = self.coarse_lon[k] as f64;
            if nlat.is_finite() && nlon.is_finite() {
                dist2(center_lat, center_lon, nlat, nlon, lon_weight)
            } else {
                0.0
            }
        };
        let mut sx2 = 0.0f64;
        for ni in [si.wrapping_sub(1), si + 1] {
            if ni < stride_x {
                sx2 = sx2.max(spacing2(sj * stride_x + ni));
            }
        }
        let mut sy2 = 0.0f64;
        for nj in [sj.wrapping_sub(1), sj + 1] {
            if nj < self.sample_ys.len() {
                sy2 = sy2.max(spacing2(nj * stride_x + si));
            }
        }
        sx2 + sy2
    }

    /// Inverse bilinear within the cell whose lower corner is (cell_x,
    /// cell_y): Newton-solve lat/lon(s, t) = query for (s, t) in [0, 1]^2.
    /// Returns `None` when the cell is degenerate/NaN, Newton wanders
    /// outside, or the solution is not (within tolerance) inside the cell.
    fn invert_cell(
        &self,
        cell_x: usize,
        cell_y: usize,
        qlat: f64,
        qlon: f64,
    ) -> Option<(f64, f64)> {
        let corner = |x: usize, y: usize| -> (f64, f64) {
            let idx = y * self.nx + x;
            (self.lat[idx] as f64, self.lon[idx] as f64)
        };
        let (lat00, lon00) = corner(cell_x, cell_y);
        let (lat10, lon10) = corner(cell_x + 1, cell_y);
        let (lat01, lon01) = corner(cell_x, cell_y + 1);
        let (lat11, lon11) = corner(cell_x + 1, cell_y + 1);
        if ![lat00, lat10, lat01, lat11, lon00, lon10, lon01, lon11]
            .iter()
            .all(|v| v.is_finite())
        {
            return None;
        }
        // Express longitudes relative to corner 00 so cells spanning the
        // dateline stay continuous.
        let lon10 = lon00 + wrap_dlon(lon10 - lon00);
        let lon01 = lon00 + wrap_dlon(lon01 - lon00);
        let lon11 = lon00 + wrap_dlon(lon11 - lon00);
        let qlon = lon00 + wrap_dlon(qlon - lon00);

        let mut s = 0.5f64;
        let mut t = 0.5f64;
        for _ in 0..NEWTON_ITERATIONS {
            let bilin = |v00: f64, v10: f64, v01: f64, v11: f64| {
                (1.0 - s) * (1.0 - t) * v00
                    + s * (1.0 - t) * v10
                    + (1.0 - s) * t * v01
                    + s * t * v11
            };
            let f_lat = bilin(lat00, lat10, lat01, lat11) - qlat;
            let f_lon = bilin(lon00, lon10, lon01, lon11) - qlon;
            let dlat_ds = (1.0 - t) * (lat10 - lat00) + t * (lat11 - lat01);
            let dlat_dt = (1.0 - s) * (lat01 - lat00) + s * (lat11 - lat10);
            let dlon_ds = (1.0 - t) * (lon10 - lon00) + t * (lon11 - lon01);
            let dlon_dt = (1.0 - s) * (lon01 - lon00) + s * (lon11 - lon10);
            let det = dlon_ds * dlat_dt - dlon_dt * dlat_ds;
            if !det.is_finite() || det.abs() < 1e-18 {
                return None; // degenerate cell
            }
            s -= (dlat_dt * f_lon - dlon_dt * f_lat) / det;
            t -= (-dlat_ds * f_lon + dlon_ds * f_lat) / det;
            if !s.is_finite()
                || !t.is_finite()
                || !(-1.0..=2.0).contains(&s)
                || !(-1.0..=2.0).contains(&t)
            {
                return None; // Newton wandered outside the neighborhood
            }
        }
        if !(-CELL_TOLERANCE..=1.0 + CELL_TOLERANCE).contains(&s)
            || !(-CELL_TOLERANCE..=1.0 + CELL_TOLERANCE).contains(&t)
        {
            return None;
        }
        Some((s.clamp(0.0, 1.0), t.clamp(0.0, 1.0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustwx_core::{GridProjection, GridShape, LatLonGrid};
    use sha2::{Digest, Sha256};
    use std::fs;
    use std::path::PathBuf;

    const NX: usize = 50;
    const NY: usize = 40;
    const LAT0: f64 = 30.0;
    const LON0: f64 = -100.0;
    const DLAT: f64 = 0.05;
    const DLON: f64 = 0.05;

    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("rw-store-grid-{}-{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Regular 50x40 grid: lat 30..32 by 0.05, lon -100..-97.55 by 0.05.
    fn regular_grid() -> LatLonGrid {
        let mut lat = Vec::with_capacity(NX * NY);
        let mut lon = Vec::with_capacity(NX * NY);
        for y in 0..NY {
            for x in 0..NX {
                lat.push((LAT0 + DLAT * y as f64) as f32);
                lon.push((LON0 + DLON * x as f64) as f32);
            }
        }
        LatLonGrid::new(GridShape::new(NX, NY).unwrap(), lat, lon).unwrap()
    }

    fn sha256_hex_of(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    fn grid_file_from(grid: &LatLonGrid) -> GridFile {
        GridFile {
            nx: grid.shape.nx,
            ny: grid.shape.ny,
            lat: grid.lat_deg.clone(),
            lon: grid.lon_deg.clone(),
            projection: None,
            hash: String::new(),
        }
    }

    #[test]
    fn file_length_cap_rejects_before_allocation() {
        let err = validate_grid_file_len(MAX_GRID_FILE_LEN + 1).unwrap_err();
        match err {
            RwStoreError::Format(message) => {
                assert!(message.contains("limit"), "unexpected message: {message}")
            }
            other => panic!("expected Format error, got {other:?}"),
        }
    }

    #[test]
    fn declared_grid_meta_cap_rejects_before_allocation() {
        let mut header = [0u8; HEADER_LEN];
        header[0..8].copy_from_slice(GRID_MAGIC);
        header[8..12].copy_from_slice(&GRID_VERSION.to_le_bytes());
        let declared = (MAX_GRID_META_LEN + 1) as u32;
        header[12..16].copy_from_slice(&declared.to_le_bytes());
        let err = parse_grid_layout(&header, HEADER_LEN as u64 + u64::from(declared)).unwrap_err();
        match err {
            RwStoreError::Format(message) => assert!(
                message.contains("meta") && message.contains("limit"),
                "unexpected message: {message}"
            ),
            other => panic!("expected Format error, got {other:?}"),
        }
    }

    #[test]
    fn open_rejects_header_layout_before_file_allocation() {
        let dir = test_dir("bad-layout");
        let path = dir.join("bad-layout.rwg");
        let mut header = [0u8; HEADER_LEN];
        header[0..8].copy_from_slice(GRID_MAGIC);
        header[8..12].copy_from_slice(&GRID_VERSION.to_le_bytes());
        header[16..24].copy_from_slice(&1u64.to_le_bytes());
        header[24..32].copy_from_slice(&1u64.to_le_bytes());
        fs::write(&path, header).unwrap();

        let err = GridFile::open(&path).unwrap_err();
        match err {
            RwStoreError::Format(message) => assert!(
                message.contains("length mismatch"),
                "unexpected message: {message}"
            ),
            other => panic!("expected Format error, got {other:?}"),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn byte_buffer_allocation_failure_is_reported() {
        let err = try_zeroed_bytes(usize::MAX, "test buffer").unwrap_err();
        match err {
            RwStoreError::Format(message) => assert!(
                message.contains("cannot allocate"),
                "unexpected message: {message}"
            ),
            other => panic!("expected Format error, got {other:?}"),
        }
    }

    #[test]
    fn grid_file_round_trips() {
        let dir = test_dir("round-trip");
        let path = dir.join("grid.rwg");
        let grid = regular_grid();
        let written_hash = write_grid(&path, &grid, Some(&GridProjection::Geographic)).unwrap();

        let file = GridFile::open(&path).unwrap();
        assert_eq!(file.nx, NX);
        assert_eq!(file.ny, NY);
        assert_eq!(file.lat.len(), NX * NY);
        assert_eq!(file.lon.len(), NX * NY);
        for (got, want) in file.lat.iter().zip(grid.lat_deg.iter()) {
            assert_eq!(got.to_bits(), want.to_bits(), "lat must be bit-exact");
        }
        for (got, want) in file.lon.iter().zip(grid.lon_deg.iter()) {
            assert_eq!(got.to_bits(), want.to_bits(), "lon must be bit-exact");
        }
        assert_eq!(file.projection, Some(GridProjection::Geographic));

        assert_eq!(written_hash.len(), 64, "sha256 hex must be 64 chars");
        assert!(
            written_hash
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "hash must be lowercase hex: {written_hash}"
        );
        assert_eq!(file.hash, written_hash, "open() must report the same hash");
        let bytes = fs::read(&path).unwrap();
        assert_eq!(
            sha256_hex_of(&bytes),
            written_hash,
            "hash must be sha256 of the final file bytes"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn grid_hash_is_stable() {
        let dir = test_dir("hash-stable");
        let grid = regular_grid();
        let hash_one = write_grid(&dir.join("one.rwg"), &grid, None).unwrap();
        let hash_two = write_grid(&dir.join("two.rwg"), &grid, None).unwrap();
        assert_eq!(hash_one, hash_two, "identical grids must hash identically");

        let mut mutated = regular_grid();
        mutated.lon_deg[123] += 0.001;
        let hash_three = write_grid(&dir.join("three.rwg"), &mutated, None).unwrap();
        assert_ne!(
            hash_one, hash_three,
            "changing one lon value must change the hash"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn locator_finds_points_on_regular_grid() {
        let file = grid_file_from(&regular_grid());
        let locator = GridLocator::build(&file);
        // (fx, fy) ground truth: exact grid points, cell centers, near edges.
        let queries: &[(f64, f64)] = &[
            (0.0, 0.0),
            (10.0, 20.0),
            (49.0, 39.0),
            (5.5, 7.5),
            (20.5, 10.5),
            (48.5, 38.5),
            (0.01, 0.01),
            (48.99, 38.99),
            (25.0, 0.02),
            (0.02, 25.0),
            (49.0, 0.5),
            (12.25, 39.0),
        ];
        let mut max_err = 0.0f64;
        for &(fx, fy) in queries {
            let lat = LAT0 + DLAT * fy;
            let lon = LON0 + DLON * fx;
            let (gx, gy) = locator
                .locate(lat, lon)
                .unwrap_or_else(|| panic!("locate returned None for fx={fx} fy={fy}"));
            let err = (gx - fx).abs().max((gy - fy).abs());
            max_err = max_err.max(err);
            assert!(
                (gx - fx).abs() < 1e-3 && (gy - fy).abs() < 1e-3,
                "query (fx={fx}, fy={fy}) located at ({gx}, {gy}), err {err}"
            );
        }
        eprintln!("regular grid locator max error: {max_err:.2e} grid cells");
    }

    const RNX: usize = 60;
    const RNY: usize = 50;

    fn rotated_lat(fx: f64, fy: f64) -> f64 {
        30.0 + 0.01 * fy + 0.002 * fx
    }

    fn rotated_lon(fx: f64, fy: f64) -> f64 {
        -100.0 + 0.012 * fx - 0.003 * fy
    }

    fn rotated_grid_file() -> GridFile {
        let mut lat = Vec::with_capacity(RNX * RNY);
        let mut lon = Vec::with_capacity(RNX * RNY);
        for y in 0..RNY {
            for x in 0..RNX {
                lat.push(rotated_lat(x as f64, y as f64) as f32);
                lon.push(rotated_lon(x as f64, y as f64) as f32);
            }
        }
        GridFile {
            nx: RNX,
            ny: RNY,
            lat,
            lon,
            projection: None,
            hash: String::new(),
        }
    }

    #[test]
    fn locator_handles_rotated_grid() {
        let file = rotated_grid_file();
        let locator = GridLocator::build(&file);
        let queries: &[(f64, f64)] = &[
            (0.0, 0.0),
            (0.5, 0.5),
            (10.25, 20.75),
            (33.4, 7.9),
            (59.0, 49.0),
            (58.5, 48.5),
            (45.1, 2.2),
            (3.7, 44.3),
        ];
        let mut max_err = 0.0f64;
        for &(fx, fy) in queries {
            let lat = rotated_lat(fx, fy);
            let lon = rotated_lon(fx, fy);
            let (gx, gy) = locator
                .locate(lat, lon)
                .unwrap_or_else(|| panic!("locate returned None for fx={fx} fy={fy}"));
            let err = (gx - fx).abs().max((gy - fy).abs());
            max_err = max_err.max(err);
            assert!(
                (gx - fx).abs() < 1e-2 && (gy - fy).abs() < 1e-2,
                "query (fx={fx}, fy={fy}) located at ({gx}, {gy}), err {err}"
            );
        }
        eprintln!("rotated grid locator max error: {max_err:.2e} grid cells");
    }

    #[test]
    fn lat_descending_derives_orientation() {
        // South-to-north (synthetic/regular): row 0 is the southernmost.
        let south_up = grid_file_from(&regular_grid());
        assert_eq!(south_up.lat_descending(), Some(false));

        // North-to-south: reverse the rows.
        let mut north_up = south_up.clone();
        for y in 0..NY {
            let src = &south_up.lat[(NY - 1 - y) * NX..(NY - y) * NX];
            north_up.lat[y * NX..(y + 1) * NX].copy_from_slice(src);
        }
        assert_eq!(north_up.lat_descending(), Some(true));

        // NaN-safe: poison the first rows and the whole first column; the
        // scan must skip NaNs and still decide from later finite pairs.
        let mut holed = grid_file_from(&regular_grid());
        for v in holed.lat[..2 * NX].iter_mut() {
            *v = f32::NAN;
        }
        for y in 0..NY {
            holed.lat[y * NX] = f32::NAN;
        }
        assert_eq!(holed.lat_descending(), Some(false));

        // Undecidable: all-NaN and constant latitude.
        let mut all_nan = grid_file_from(&regular_grid());
        all_nan.lat.fill(f32::NAN);
        assert_eq!(all_nan.lat_descending(), None);
        let mut flat = grid_file_from(&regular_grid());
        flat.lat.fill(42.0);
        assert_eq!(flat.lat_descending(), None);
    }

    /// Probe the REAL ingested HRRR grid: definitively answer whether row 0
    /// is south or north. Run with:
    /// `cargo test -p rw-store real_hrrr_grid -- --ignored --nocapture`
    #[test]
    #[ignore = "requires the real store at C:/Users/drew/rusty-weather/store"]
    fn real_hrrr_grid_row_order() {
        let path = Path::new("C:/Users/drew/rusty-weather/store/hrrr/20260608_00z/grid.rwg");
        let grid = GridFile::open(path).expect("real grid.rwg readable");
        let (nx, ny) = (grid.nx, grid.ny);
        let first = grid.lat[0];
        let last = grid.lat[(ny - 1) * nx];
        eprintln!("real HRRR grid {nx}x{ny}");
        eprintln!("lat[row 0,    col 0] = {first}");
        eprintln!("lat[row ny-1, col 0] = {last}");
        eprintln!("lon[row 0,    col 0] = {}", grid.lon[0]);
        eprintln!("lat_descending() = {:?}", grid.lat_descending());
        let derived = grid.lat_descending().expect("real grid must decide");
        assert_eq!(
            derived,
            first > last,
            "helper must agree with the corner latitudes"
        );
    }

    #[test]
    fn locator_rejects_far_outside_points() {
        let file = grid_file_from(&regular_grid());
        let locator = GridLocator::build(&file);
        // Grid covers lat 30..31.95, lon -100..-97.55; both queries are far out.
        assert_eq!(
            locator.locate(LAT0 + 30.0, LON0 + 1.0),
            None,
            "point 30 degrees north of the grid must be rejected"
        );
        assert_eq!(
            locator.locate(LAT0 + 1.0, LON0 - 40.0),
            None,
            "point 40 degrees west of the grid must be rejected"
        );
    }
}
