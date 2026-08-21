//! LANE 2.  The domain sampler (`_DomainSampler` in
//! `gpuwm/static/build.py`): extended grid (+HALO), float32-twin
//! sampling coordinates with the compiler-band reconciliations, source
//! windows, pixel-to-cell binning (Fortran `nint` accumulation rule),
//! grid-cell means, categorical fractions, and the mandatory-coverage
//! gate that renders `gpuwm-geog-source-coverage-v1` receipts.
//!
//! Every float32 decision in the Python is stencil-SELECTING and part
//! of the byte-parity contract: the f32 `cell_coords` path for
//! sub-kilometre grids (WPS regular_ll default-REAL operation order),
//! the `yi_lower` / tile-boundary `use_lower` reconciliation, the
//! east-nudge on `_lon_boundary_band`, the half-cell 5e-5 snap in
//! `pixel_cells`, the f32 reciprocal in categorical normalization, and
//! the f32 plane normalization of continuous category sources.
//!
//! Mesh model: [`SamplerMesh`] carries the twin/public transform
//! OUTPUTS (extended-grid lat/lon in both precisions reduced to the
//! sampling arrays + band masks).  [`SamplerMesh::from_twin_outputs`]
//! is the pure assembly of `_DomainSampler.__init__`'s float logic; the
//! transforms themselves are lane 1's (`projection::wps32`), consumed
//! by [`DomainSampler::new`].  Tests drive the mesh with fixture arrays
//! extracted from the Python.
//!
//! Determinism: parallel loops are strictly elementwise (rayon over
//! points); every ACCUMULATION (`accum_mean`, category counts) is
//! sequential in numpy's `bincount` element order so f64 sums carry the
//! Python's exact association.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use crate::error::{Result, StaticError};
use crate::geog::{GeogDataset, GeogWindow, SourceType};
use crate::interp::{interp_seq, InterpOp, WindowView};
use crate::projection::{wps32_for, ProjectedGrid, ProjectionKind};
use crate::types::{Grid2, Stack3};
use crate::{GCELL_RATIO, M_PER_DEG};

// -- float helpers (numpy semantics) ------------------------------------

/// `np.spacing` for float32: the ULP in the away-from-zero direction
/// (sign of `x`), as numpy defines it.
pub(crate) fn spacing32(x: f32) -> f32 {
    if x >= 0.0 { x.next_up() - x } else { x.next_down() - x }
}

/// Mirror GNU WPS's optimized single-precision longitude result
/// (`_DomainSampler._geogrid_longitude`): band the compiler's three
/// evaluation regimes against the public float64 transform and step the
/// float32 value east accordingly.
pub fn geogrid_longitude(lon32: &[f32], lon64: &[f64]) -> Vec<f32> {
    lon32
        .iter()
        .zip(lon64.iter())
        .map(|(&c, &d)| {
            let p = d as f32;
            let ci = c.to_bits() as i32;
            let pi = p.to_bits() as i32;
            let ulp = (spacing32(p) as f64).abs();
            let frac = if ulp != 0.0 { (d - p as f64) / ulp } else { 0.0 };
            let band = ci.wrapping_sub(pi);
            let mut steps: u8 = 4;
            if band <= -3 || (band == -2 && frac < -0.05) {
                steps = 0;
            }
            if band >= 3 || (band == 2 && frac >= -0.05) {
                steps = 8;
            }
            let mut out = c;
            for _ in 0..steps {
                out = out.next_up();
            }
            out
        })
        .collect()
}

// -- the sampler mesh ---------------------------------------------------

/// The extended-grid longitude sampling array: float32 for the
/// sub-kilometre Lambert path (`_geogrid_longitude` output), float64
/// everywhere else -- exactly the dtype split `self.lon_e` carries.
#[derive(Debug, Clone)]
pub enum LonE {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// Sampling-coordinate mesh for one extended domain (the float state
/// `_DomainSampler.__init__` derives from the twin + public
/// transforms).
#[derive(Debug, Clone)]
pub struct SamplerMesh {
    pub nxe: usize,
    pub nye: usize,
    /// Extended-grid mass-point latitude through the f32 twin
    /// (sub-kilometre Lambert: already shifted one ULP down).
    pub lat_e: Vec<f32>,
    /// One ULP below `lat_e` (`_lat_lower_e`).
    pub lat_lower_e: Vec<f32>,
    pub lon_e: LonE,
    /// `_lon_boundary_band` (Lambert only; empty bands elsewhere).
    pub lon_boundary_band: Vec<bool>,
    /// `_lat_integer_band` (Lambert only).
    pub lat_integer_band: Vec<bool>,
    /// Cell-corner mesh `(nye+1, nxe+1)`: latitude via the f32 twin,
    /// longitude via the public f64 transform (window bounds only).
    pub lat_c: Vec<f32>,
    pub lon_c: Vec<f64>,
}

impl SamplerMesh {
    /// Assemble the mesh from raw transform outputs -- the pure float
    /// logic of `_DomainSampler.__init__` after the twin/public
    /// transforms have produced the extended and corner meshes.
    ///
    /// `lat32`/`lon32` are the f32 twin outputs (nudges included, as
    /// the twin applies them); `lat64`/`lon64` the public f64 outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn from_twin_outputs(
        nxe: usize,
        nye: usize,
        lat32: Vec<f32>,
        lon32: Vec<f32>,
        lat64: &[f64],
        lon64: Vec<f64>,
        lat_c: Vec<f32>,
        lon_c: Vec<f64>,
        is_lambert: bool,
        dx: f64,
    ) -> Result<SamplerMesh> {
        let n = nxe * nye;
        let corner = (nxe + 1) * (nye + 1);
        for (name, len) in [
            ("lat32", lat32.len()),
            ("lon32", lon32.len()),
            ("lat64", lat64.len()),
            ("lon64", lon64.len()),
        ] {
            if len != n {
                return Err(StaticError::Invalid(format!(
                    "sampler mesh input {name} has {len} points, \
                     expected {n}"
                )));
            }
        }
        for (name, len) in [("lat_c", lat_c.len()), ("lon_c", lon_c.len())]
        {
            if len != corner {
                return Err(StaticError::Invalid(format!(
                    "sampler corner mesh {name} has {len} points, \
                     expected {corner}"
                )));
            }
        }
        let subkm = dx < 1000.0;
        let mut lat_e = lat32;
        let mut lat_lower_e: Vec<f32> =
            lat_e.iter().map(|v| v.next_down()).collect();
        if is_lambert && subkm {
            // One of the two scalar-libm ULPs documented in
            // `_WpsLambert32.ij_to_latlon` is absorbed when geogrid
            // initializes a nest from its centre.
            lat_e = lat_lower_e;
            lat_lower_e = lat_e.iter().map(|v| v.next_down()).collect();
        }
        let lon_e = if is_lambert && subkm {
            LonE::F32(geogrid_longitude(&lon32, &lon64))
        } else {
            LonE::F64(lon64.clone())
        };
        let (lon_boundary_band, lat_integer_band) = if is_lambert {
            let mut lon_band = vec![false; n];
            let mut lat_band = vec![false; n];
            for k in 0..n {
                let lon_public = lon64[k] as f32;
                let lon_ulp = (spacing32(lon_public) as f64).abs();
                let lon_frac = if lon_ulp != 0.0 {
                    (lon64[k] - lon_public as f64) / lon_ulp
                } else {
                    0.0
                };
                let band = (lon32[k].to_bits() as i32)
                    .wrapping_sub(lon_public.to_bits() as i32);
                lon_band[k] = band.wrapping_abs() == 2
                    && lon_frac >= -0.15
                    && lon_frac <= -0.05;

                let lat_public = lat64[k] as f32;
                let lat_ulp = (spacing32(lat_public) as f64).abs();
                let lat_frac = if lat_ulp != 0.0 {
                    (lat64[k] - lat_public as f64) / lat_ulp
                } else {
                    0.0
                };
                let band = (lat_lower_e[k].to_bits() as i32)
                    .wrapping_sub(lat_public.to_bits() as i32);
                lat_band[k] =
                    band == 4 && lat_frac >= 0.38 && lat_frac <= 0.41;
            }
            (lon_band, lat_band)
        } else {
            (vec![false; n], vec![false; n])
        };
        Ok(SamplerMesh {
            nxe,
            nye,
            lat_e,
            lat_lower_e,
            lon_e,
            lon_boundary_band,
            lat_integer_band,
            lat_c,
            lon_c,
        })
    }
}

// -- pixel binning ------------------------------------------------------

/// Cells-cache key: the Python `_cells_cache` tuple
/// `(idx.dx, idx.dy, known_x, known_y, known_lat, known_lon,
///   win.x0, win.y0, win.raw.shape)` with floats keyed by bit pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CellsKey {
    pub dx: u64,
    pub dy: u64,
    pub known_x: u64,
    pub known_y: u64,
    pub known_lat: u64,
    pub known_lon: u64,
    pub x0: i64,
    pub y0: i64,
    pub nz: usize,
    pub ny: usize,
    pub nx: usize,
}

impl CellsKey {
    pub fn for_window(ds: &GeogDataset, win: &GeogWindow) -> CellsKey {
        let idx = &ds.index;
        CellsKey {
            dx: idx.dx.to_bits(),
            dy: idx.dy.to_bits(),
            known_x: idx.known_x.to_bits(),
            known_y: idx.known_y.to_bits(),
            known_lat: idx.known_lat.to_bits(),
            known_lon: idx.known_lon.to_bits(),
            x0: win.x0,
            y0: win.y0,
            nz: win.nz,
            ny: win.ny,
            nx: win.nx,
        }
    }
}

/// The pure `nint` binning step of `pixel_cells`: model-grid
/// coordinates -> flat extended-cell index (-1 outside), including the
/// half-cell 5e-5 snap.  Elementwise, rayon-parallel, order-free.
pub fn bin_grid_coords(
    gx: &[f64],
    gy: &[f64],
    nxe: usize,
    nye: usize,
    halo: usize,
) -> Vec<i64> {
    gx.par_iter()
        .zip(gy.par_iter())
        .map(|(&gx0, &gy0)| {
            if !gx0.is_finite() || !gy0.is_finite() {
                return -1;
            }
            let mut gx = gx0;
            let mut gy = gy0;
            // Default-REAL Lambert math can leave an analytically
            // half-cell coordinate a few 1e-5 away from the boundary
            // before NINT.
            let hx = gx.floor() + 0.5;
            let hy = gy.floor() + 0.5;
            if (gx - hx).abs() < 5e-5 {
                gx = hx;
            }
            if (gy - hy).abs() < 5e-5 {
                gy = hy;
            }
            let ei = (gx + 0.5).floor() as i64 + (halo as i64 - 1);
            let ej = (gy + 0.5).floor() as i64 + (halo as i64 - 1);
            if ei >= 0
                && ei < nxe as i64
                && ej >= 0
                && ej < nye as i64
            {
                ej * nxe as i64 + ei
            } else {
                -1
            }
        })
        .collect()
}

/// Extended-grid sampling coordinates for one window, in the precision
/// WPS uses for the grid (`cell_coords` dtype split).
#[derive(Debug, Clone)]
pub enum CoordArray {
    F32 { xi: Vec<f32>, yi: Vec<f32> },
    F64 { xi: Vec<f64>, yi: Vec<f64> },
}

// -- the sampler --------------------------------------------------------

/// The sampler for one (extended) domain.
///
/// NOTE for the assembler: the skeleton declared `grid` as a bare
/// reference; it is optional here so mesh-fixture tests can drive the
/// sampler before lane 1's transforms land.  `DomainSampler::new` is
/// unchanged and remains the only production constructor.
pub struct DomainSampler<'g> {
    pub grid: Option<&'g ProjectedGrid>,
    /// Grid spacing (m), the gcell-ratio and precision selector.
    pub dx: f64,
    pub halo: usize,
    /// Mass-grid dims (e_we-1, e_sn-1) and extended dims.
    pub nx: usize,
    pub ny: usize,
    pub nxe: usize,
    pub nye: usize,
    pub mesh: SamplerMesh,
    /// Keep only the last mapping, as the Python does.
    cells_cache: Mutex<Option<(CellsKey, Arc<Vec<i64>>)>>,
    /// Test injection: precomputed `pixel_cells` outputs keyed like the
    /// cache.  Empty in production.
    pub(crate) fixture_cells: BTreeMap<CellsKey, Arc<Vec<i64>>>,
}

impl<'g> DomainSampler<'g> {
    /// Build the sampler for a projected grid: f32 twin + public f64
    /// transforms over the extended and corner meshes, assembled by
    /// [`SamplerMesh::from_twin_outputs`].  LANE 2 (transforms are
    /// lane 1's; this wiring goes live when they land).
    pub fn new(grid: &'g ProjectedGrid, halo: usize) -> Result<Self> {
        let mut twin = wps32_for(grid)?;
        let nx = (grid.spec.e_we - 1) as usize;
        let ny = (grid.spec.e_sn - 1) as usize;
        let nxe = nx + 2 * halo;
        let nye = ny + 2 * halo;
        if grid.spec.dx < 1000.0 {
            // WPS locates nests from their mass-grid centre; the public
            // float64 projection already carries the resulting pole.
            twin.adopt_public_pole(grid);
        }
        let n = nxe * nye;
        let mut lat32 = Vec::with_capacity(n);
        let mut lon32 = Vec::with_capacity(n);
        let mut lat64 = Vec::with_capacity(n);
        let mut lon64 = Vec::with_capacity(n);
        for j in 0..nye {
            let y = (1 - halo as i64 + j as i64) as f64;
            for i in 0..nxe {
                let x = (1 - halo as i64 + i as i64) as f64;
                let (la32, lo32) = twin.ij_to_latlon32(x as f32, y as f32);
                lat32.push(la32);
                lon32.push(lo32);
                let (la64, lo64) = grid.ij_to_latlon(x, y);
                lat64.push(la64);
                lon64.push(lo64);
            }
        }
        let mut lat_c = Vec::with_capacity((nxe + 1) * (nye + 1));
        let mut lon_c = Vec::with_capacity((nxe + 1) * (nye + 1));
        for j in 0..=nye {
            let y = (0.5 - halo as f64) + j as f64;
            for i in 0..=nxe {
                let x = (0.5 - halo as f64) + i as f64;
                let (la32, _) = twin.ij_to_latlon32(x as f32, y as f32);
                lat_c.push(la32);
                let (_, lo64) = grid.ij_to_latlon(x, y);
                lon_c.push(lo64);
            }
        }
        let mesh = SamplerMesh::from_twin_outputs(
            nxe,
            nye,
            lat32,
            lon32,
            &lat64,
            lon64,
            lat_c,
            lon_c,
            matches!(grid.spec.kind, ProjectionKind::Lambert),
            grid.spec.dx,
        )?;
        Ok(DomainSampler {
            grid: Some(grid),
            dx: grid.spec.dx,
            halo,
            nx,
            ny,
            nxe,
            nye,
            mesh,
            cells_cache: Mutex::new(None),
            fixture_cells: BTreeMap::new(),
        })
    }

    /// Assemble a sampler from parts (mesh fixtures; also the seam the
    /// assembler can reuse).  `grid` may be absent only when every
    /// window the build touches has fixture cells.
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(test), allow(dead_code))] // production path is `new`
    pub(crate) fn from_parts(
        grid: Option<&'g ProjectedGrid>,
        dx: f64,
        halo: usize,
        nx: usize,
        ny: usize,
        mesh: SamplerMesh,
        fixture_cells: BTreeMap<CellsKey, Arc<Vec<i64>>>,
    ) -> Result<Self> {
        let nxe = nx + 2 * halo;
        let nye = ny + 2 * halo;
        if mesh.nxe != nxe || mesh.nye != nye {
            return Err(StaticError::Invalid(format!(
                "sampler mesh is {}x{}, domain wants {nye}x{nxe}",
                mesh.nye, mesh.nxe
            )));
        }
        Ok(DomainSampler {
            grid,
            dx,
            halo,
            nx,
            ny,
            nxe,
            nye,
            mesh,
            cells_cache: Mutex::new(None),
            fixture_cells,
        })
    }

    /// Source-to-grid resolution ratio (`res_ratio`).
    pub fn res_ratio(&self, ds: &GeogDataset) -> f64 {
        self.dx / (M_PER_DEG * ds.index.dx.abs())
    }

    /// Read the source window covering the extended grid + margin
    /// (`window`).  LANE 2.
    pub fn window(&self, ds: &GeogDataset, margin: i64) -> Result<GeogWindow> {
        let mut xs = Vec::with_capacity(self.mesh.lat_c.len());
        let mut ys = Vec::with_capacity(self.mesh.lat_c.len());
        for (&lat, &lon) in self.mesh.lat_c.iter().zip(&self.mesh.lon_c) {
            let (x, y) = ds.latlon_to_xy(lat as f64, lon);
            xs.push(x);
            ys.push(y);
        }
        if ds.wraps_x {
            let xmin = xs.iter().cloned().fold(f64::INFINITY, f64::min);
            let xmax = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if xmax - xmin > ds.nx_global as f64 / 2.0 {
                for x in &mut xs {
                    if *x < ds.nx_global as f64 / 2.0 {
                        *x += ds.nx_global as f64;
                    }
                }
            }
        }
        let xmin = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let xmax = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ymin = ys.iter().cloned().fold(f64::INFINITY, f64::min);
        let ymax = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let x0 = xmin.floor() as i64 - margin;
        let x1 = xmax.ceil() as i64 + margin;
        let y0 = (ymin.floor() as i64 - margin).max(1);
        let y1 = (ymax.ceil() as i64 + margin).min(ds.ny_global);
        ds.read_window(x0, x1, y0, y1)
    }

    /// Prove every source cell of a mandatory window is tiled
    /// (`require_source_coverage`); Ok carries the PASS receipt JSON,
    /// Err the refusal naming missing origins.  LANE 2.
    pub fn require_source_coverage(
        &self,
        ds: &GeogDataset,
        win: &GeogWindow,
        field: &str,
    ) -> Result<String> {
        let expected = win.ny * win.nx;
        let Some(coverage) = win.coverage.as_ref().filter(|c| c.len() == expected)
        else {
            return Err(StaticError::Invalid(format!(
                "WPS GEOG coverage metadata for mandatory field '{field}' \
                 has shape {}; expected ({}, {})",
                win.coverage
                    .as_ref()
                    .map(|c| format!("len {}", c.len()))
                    .unwrap_or_else(|| "None".to_string()),
                win.ny,
                win.nx
            )));
        };
        let required_cells = coverage.len();
        let covered_cells = coverage.iter().filter(|&&c| c).count();
        let missing_cells = required_cells - covered_cells;
        let extent = ds.extent_mask(win.x0, win.x1(), win.y0, win.y1());
        let missing_tile_cells = coverage
            .iter()
            .zip(&extent)
            .filter(|&(&c, &e)| !c && e)
            .count();
        let outside_extent_cells = coverage
            .iter()
            .zip(&extent)
            .filter(|&(&c, &e)| !c && !e)
            .count();
        let required_origins =
            ds.required_tile_origins(win.x0, win.x1(), win.y0, win.y1())?;

        if missing_cells > 0 {
            let missing_tiles =
                ds.missing_tiles(win.x0, win.x1(), win.y0, win.y1())?;
            let first = coverage
                .iter()
                .position(|&c| !c)
                .expect("missing_cells > 0 implies an uncovered cell");
            let mut source_x = win.x0 + (first % win.nx) as i64;
            let source_y = win.y0 + (first / win.nx) as i64;
            if ds.wraps_x {
                source_x = (source_x - 1).rem_euclid(ds.nx_global) + 1;
            }
            let preview: Vec<String> = missing_tiles
                .iter()
                .take(16)
                .map(|(x, y)| format!("[{x}, {y}]"))
                .collect();
            let suffix = if missing_tiles.len() > 16 { "..." } else { "" };
            return Err(StaticError::Missing(format!(
                "WPS GEOG mandatory source coverage failed for \
                 field '{field}' in {}: covered \
                 {covered_cells}/{required_cells} \
                 source cells; first uncovered source index=(x={source_x}, \
                 y={source_y}); missing_tile_cells={missing_tile_cells}; \
                 outside_extent_cells={outside_extent_cells}; \
                 missing_tile_origins=[{}]{suffix}; source_window=x={}..{},\
                 y={}..{}; declared_sparse={}. A sparse staging declaration \
                 does not permit absent tiles required by a model domain.",
                resolved_path_string(&ds.path),
                preview.join(", "),
                win.x0,
                win.x1(),
                win.y0,
                win.y1(),
                if ds.declared_sparse { "True" } else { "False" }
            )));
        }

        let mut required_tiles = Vec::with_capacity(required_origins.len());
        for origin in &required_origins {
            let Some(path) = ds.tiles.get(origin) else {
                return Err(StaticError::Invalid(format!(
                    "coverage passed but required GEOG tile ({}, {}) is \
                     absent",
                    origin.0, origin.1
                )));
            };
            let bytes = std::fs::metadata(path)?.len();
            required_tiles.push(RequiredTile {
                origin: [origin.0, origin.1],
                relative_path: path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                bytes,
            });
        }
        let receipt = CoverageReceipt {
            schema: "gpuwm-geog-source-coverage-v1",
            status: "PASS",
            field: field.to_string(),
            dataset: resolved_path_string(&ds.path),
            declared_sparse: ds.declared_sparse,
            source_geometry: SourceGeometry {
                nx_global: ds.nx_global,
                ny_global: ds.ny_global,
                wraps_x: ds.wraps_x,
                extent_basis: ds.extent_basis.to_string(),
                tile_inventory_bounds: [
                    ds.tile_inventory_bounds.0,
                    ds.tile_inventory_bounds.1,
                    ds.tile_inventory_bounds.2,
                    ds.tile_inventory_bounds.3,
                ],
            },
            source_window: SourceWindow {
                x_start: win.x0,
                x_end: win.x1(),
                y_start: win.y0,
                y_end: win.y1(),
            },
            required_cells,
            covered_cells,
            missing_tile_cells: 0,
            outside_extent_cells: 0,
            coverage_fraction: 1.0,
            required_tile_count: required_tiles.len(),
            required_tiles,
        };
        serde_json::to_string(&receipt).map_err(|err| {
            StaticError::Invalid(format!(
                "coverage receipt serialization failed: {err}"
            ))
        })
    }

    /// Flat extended-cell index per window pixel, -1 outside
    /// (`pixel_cells`).  LANE 2, rayon row-chunked (elementwise).
    pub fn pixel_cells(
        &self,
        ds: &GeogDataset,
        win: &GeogWindow,
    ) -> Result<Vec<i64>> {
        self.cells(ds, win).map(|arc| (*arc).clone())
    }

    pub(crate) fn cells(
        &self,
        ds: &GeogDataset,
        win: &GeogWindow,
    ) -> Result<Arc<Vec<i64>>> {
        let key = CellsKey::for_window(ds, win);
        if let Some((cached_key, cached)) = self
            .cells_cache
            .lock()
            .expect("cells cache poisoned")
            .as_ref()
        {
            if *cached_key == key {
                return Ok(cached.clone());
            }
        }
        let flat = if let Some(fixture) = self.fixture_cells.get(&key) {
            fixture.clone()
        } else {
            let Some(grid) = self.grid else {
                return Err(StaticError::Invalid(
                    "sampler has no grid transform for pixel binning \
                     (a fixture-driven sampler covers only the windows \
                     its fixtures name)"
                        .to_string(),
                ));
            };
            let nyw = win.ny;
            let nxw = win.nx;
            let mut coords = vec![(0.0f64, 0.0f64); nyw * nxw];
            coords
                .par_iter_mut()
                .enumerate()
                .for_each(|(k, slot)| {
                    let i = k % nxw;
                    let j = k / nxw;
                    let (lat, lon) = ds.xy_to_latlon(
                        (win.x0 + i as i64) as f64,
                        (win.y0 + j as i64) as f64,
                    );
                    *slot = grid.latlon_to_ij(lat, lon);
                });
            let gx: Vec<f64> = coords.iter().map(|c| c.0).collect();
            let gy: Vec<f64> = coords.iter().map(|c| c.1).collect();
            Arc::new(bin_grid_coords(&gx, &gy, self.nxe, self.nye, self.halo))
        };
        *self.cells_cache.lock().expect("cells cache poisoned") =
            Some((key, flat.clone()));
        Ok(flat)
    }

    /// Grid-cell mean accumulation (`accum_mean`): sequential in
    /// element order (numpy `bincount`), NaN where no pixel landed.
    pub fn accum_mean(&self, flat: &[i64], vals: &[f64]) -> Grid2 {
        let n = self.nxe * self.nye;
        let mut sums = vec![0.0f64; n];
        let mut cnt = vec![0i64; n];
        for (&cell, &v) in flat.iter().zip(vals.iter()) {
            if cell >= 0 && !v.is_nan() {
                let cell = cell as usize;
                sums[cell] += v;
                cnt[cell] += 1;
            }
        }
        let data = sums
            .iter()
            .zip(&cnt)
            .map(|(&s, &c)| if c > 0 { s / c as f64 } else { f64::NAN })
            .collect();
        Grid2 {
            ny: self.nye,
            nx: self.nxe,
            data,
        }
    }

    /// Extended-grid mass points in DATASET source coordinates
    /// (`cell_coords`): plain f64 for >= 1 km grids, WPS default-REAL
    /// f32 with the compiler-band reconciliations below 1 km.
    ///
    /// Dataset coordinates, never window coordinates.  A wrapping
    /// source used to have every point shifted into the window's frame
    /// here (`xi + nx_global` below the window origin) and shifted
    /// straight back by `interp_tile_sequence`; in floating point that
    /// round trip is LOSSY, so a cell sampled through a seam-crossing
    /// window read the source at slightly different coordinates than
    /// the same cell sampled through a window that did not cross the
    /// seam -- two builds of the same ground, two answers.  That is
    /// what refused every relocation of the 2026-08-20 prepared
    /// moving-nest run.  The one consumer that indexes the window
    /// raster (the categorical empty-cell fallback) re-frames in
    /// INTEGER index space, which is exact.  `win` stays in the
    /// signature for parity with the Python transcription and is
    /// deliberately not consulted.
    pub fn cell_coords(
        &self,
        ds: &GeogDataset,
        win: &GeogWindow,
    ) -> CoordArray {
        let _ = win;
        let n = self.nxe * self.nye;
        if self.dx >= 1000.0 {
            let mut xi = Vec::with_capacity(n);
            let mut yi = Vec::with_capacity(n);
            for k in 0..n {
                let lat = self.mesh.lat_e[k] as f64;
                let lon = match &self.mesh.lon_e {
                    LonE::F64(v) => v[k],
                    LonE::F32(v) => v[k] as f64,
                };
                let (x, y) = ds.latlon_to_xy(lat, lon);
                xi.push(x);
                yi.push(y);
            }
            return CoordArray::F64 { xi, yi };
        }

        // WPS regular_ll map state and operands are default REAL.
        // Preserve its operation ordering: subtract, divide, add, then
        // one wrap check.
        let idx = &ds.index;
        let known_lon = idx.known_lon as f32;
        let known_lat = idx.known_lat as f32;
        let dx32 = idx.dx as f32;
        let dy32 = idx.dy as f32;
        let known_x = idx.known_x as f32;
        let known_y = idx.known_y as f32;
        let tile_y = idx.tile_y as f32;
        let mut xi = Vec::with_capacity(n);
        let mut yi = Vec::with_capacity(n);
        for k in 0..n {
            let lon32 = match &self.mesh.lon_e {
                LonE::F32(v) => v[k],
                LonE::F64(v) => v[k] as f32,
            };
            let dlon = lon32 - known_lon;
            let dlat = self.mesh.lat_e[k] - known_lat;
            let mut x = dlon / dx32 + known_x;
            let mut y = dlat / dy32 + known_y;
            let y_lower =
                (self.mesh.lat_lower_e[k] - known_lat) / dy32 + known_y;

            // At high zoom, reconcile only the compiler bands that
            // straddle an interpolation-control boundary.
            let yint = y.round_ties_even();
            let mut use_lower = self.mesh.lat_integer_band[k]
                && y > yint
                && y_lower < yint
                && (y - yint) < 0.002f32;
            let ytile =
                ((y - 0.5f32) / tile_y + 0.5f32).floor() * tile_y + 0.5f32;
            use_lower |=
                y > ytile && y_lower < ytile && (y - ytile) < 0.002f32;
            if use_lower {
                y = y_lower;
            }

            let xint = x.round_ties_even();
            let xdist = xint - x;
            let use_east = self.mesh.lon_boundary_band[k]
                && xdist > 0.0f32
                && xdist < 0.0045f32;
            if use_east {
                x = xint.next_up();
            }
            if ds.wraps_x {
                if x < 0.5f32 {
                    x += ds.nx_global as f32;
                }
                if x >= ds.nx_global as f32 + 0.5f32 {
                    x -= ds.nx_global as f32;
                }
            }
            xi.push(x);
            yi.push(y);
        }
        CoordArray::F32 { xi, yi }
    }

    /// Interpolate points within the one native tile selected by WPS
    /// (`_interp_tile_sequence`).  `todo` indexes the mesh, row-major.
    pub(crate) fn interp_tile_sequence(
        &self,
        ds: &GeogDataset,
        z: usize,
        coords: &CoordArray,
        todo: &[usize],
        seq: &[InterpOp],
    ) -> Result<Vec<f64>> {
        let tx = ds.index.tile_x;
        let ty = ds.index.tile_y;
        // Native-dtype wrap + tile-origin selection, then f64 points
        // for the interpolators (which widen exactly as numpy does).
        let mut xx = Vec::with_capacity(todo.len());
        let mut yy = Vec::with_capacity(todo.len());
        let mut origin = Vec::with_capacity(todo.len());
        match coords {
            CoordArray::F32 { xi, yi } => {
                let hi = ((ds.nx_global as f64) + 0.5) as f32;
                for &k in todo {
                    let mut x = xi[k];
                    let y = yi[k];
                    if ds.wraps_x {
                        if x >= hi {
                            x -= ds.nx_global as f32;
                        }
                        if x < 0.5f32 {
                            x += ds.nx_global as f32;
                        }
                    }
                    let xs = ((x - 0.5f32) / (tx as f32)).floor() as i64
                        * tx
                        + 1;
                    let ys = ((y - 0.5f32) / (ty as f32)).floor() as i64
                        * ty
                        + 1;
                    origin.push((xs, ys));
                    xx.push(x as f64);
                    yy.push(y as f64);
                }
            }
            CoordArray::F64 { xi, yi } => {
                let txf = tx as f32 as f64;
                let tyf = ty as f32 as f64;
                for &k in todo {
                    let mut x = xi[k];
                    let y = yi[k];
                    if ds.wraps_x {
                        if x >= ds.nx_global as f64 + 0.5 {
                            x -= ds.nx_global as f64;
                        }
                        if x < 0.5 {
                            x += ds.nx_global as f64;
                        }
                    }
                    let xs = ((x - 0.5) / txf).floor() as i64 * tx + 1;
                    let ys = ((y - 0.5) / tyf).floor() as i64 * ty + 1;
                    origin.push((xs, ys));
                    xx.push(x);
                    yy.push(y);
                }
            }
        }
        let mut groups: BTreeMap<(i64, i64), Vec<usize>> = BTreeMap::new();
        for (t, &o) in origin.iter().enumerate() {
            groups.entry(o).or_default().push(t);
        }
        let mut out = vec![f64::NAN; todo.len()];
        for ((xs, ys), picks) in groups {
            let Some(tile) = ds.read_tile_window(xs, ys)? else {
                continue;
            };
            let vals = tile.values(z);
            let view = WindowView {
                ny: tile.ny,
                nx: tile.nx,
                vals: &vals,
                x0: tile.x0 as f64,
                y0: tile.y0 as f64,
            };
            let px: Vec<f64> = picks.iter().map(|&t| xx[t]).collect();
            let py: Vec<f64> = picks.iter().map(|&t| yy[t]).collect();
            let got = interp_seq(seq, &view, &px, &py)?;
            for (&t, v) in picks.iter().zip(got) {
                out[t] = v;
            }
        }
        Ok(out)
    }

    /// One continuous level on the extended grid (`continuous`):
    /// grid-cell average when fine enough, interp sequence fallback,
    /// then fill.  LANE 2.
    pub fn continuous(
        &self,
        ds: &GeogDataset,
        win: &GeogWindow,
        z: usize,
        seq: &[InterpOp],
        fill: f64,
        gcell: bool,
        active: Option<&[bool]>,
    ) -> Result<Grid2> {
        let n = self.nxe * self.nye;
        let vals = win.values(z);
        let mut out = if gcell && self.res_ratio(ds) >= GCELL_RATIO {
            let flat = self.cells(ds, win)?;
            self.accum_mean(&flat, &vals)
        } else {
            Grid2::filled(self.nye, self.nxe, f64::NAN)
        };
        if let Some(active) = active {
            if active.len() != n {
                return Err(StaticError::Invalid(format!(
                    "active mask has {} cells, expected {n}",
                    active.len()
                )));
            }
            for (slot, &keep) in out.data.iter_mut().zip(active) {
                if !keep {
                    *slot = fill;
                }
            }
        }
        let todo: Vec<usize> = (0..n)
            .filter(|&k| {
                out.data[k].is_nan()
                    && active.map_or(true, |a| a[k])
            })
            .collect();
        if !todo.is_empty() {
            let coords = self.cell_coords(ds, win);
            let got =
                self.interp_tile_sequence(ds, z, &coords, &todo, seq)?;
            for (&k, v) in todo.iter().zip(got) {
                out.data[k] = v;
            }
        }
        for slot in &mut out.data {
            if slot.is_nan() {
                *slot = fill;
            }
        }
        Ok(out)
    }

    /// Category fractions `(ncat, nye, nxe)` (`categorical`), both the
    /// categorical-pixel accumulation and the continuous-plane
    /// normalize path, with the empty-cell nearest fallback.  LANE 2.
    pub fn categorical(
        &self,
        ds: &GeogDataset,
        win: &GeogWindow,
        fractional_gcell: bool,
    ) -> Result<Stack3> {
        let idx = &ds.index;
        let (Some(cmin), Some(cmax)) = (idx.category_min, idx.category_max)
        else {
            return Err(StaticError::Invalid(
                "category source requires a valid \
                 category_min/category_max"
                    .to_string(),
            ));
        };
        if cmax < cmin {
            return Err(StaticError::Invalid(
                "category source requires a valid \
                 category_min/category_max"
                    .to_string(),
            ));
        }
        let ncat = (cmax - cmin + 1) as usize;
        let n = self.nxe * self.nye;

        if idx.kind == SourceType::Continuous {
            if idx.nz() as usize != ncat {
                return Err(StaticError::Invalid(format!(
                    "continuous category source declares {} z planes \
                     for {ncat} categories ({cmin}..{cmax})",
                    idx.nz()
                )));
            }
            // WPS holds both the planes and their per-cell sum in
            // default REAL.  Cells with no valid category support
            // remain all-zero.
            let mut planes32: Vec<Vec<f32>> = Vec::with_capacity(ncat);
            for z in 0..ncat {
                let plane = self.continuous(
                    ds,
                    win,
                    z,
                    &[InterpOp::FourPt],
                    0.0,
                    fractional_gcell,
                    None,
                )?;
                planes32
                    .push(plane.data.iter().map(|&v| v as f32).collect());
            }
            let mut total = vec![0.0f32; n];
            for plane in &planes32 {
                for (t, &v) in total.iter_mut().zip(plane.iter()) {
                    *t += v;
                }
            }
            // The first plane seeds the sum (numpy reduce over axis 0
            // starts from the first plane, not from zero).
            // NOTE: starting from zeros then adding plane 0 gives the
            // same bits except a -0.0 seed, which cannot arise here
            // (fractions are >= 0); kept simple.
            let mut data = vec![0.0f64; ncat * n];
            for (z, plane) in planes32.iter().enumerate() {
                for k in 0..n {
                    let frac = if total[k] > 0.0f32 {
                        plane[k] / total[k]
                    } else {
                        0.0f32
                    };
                    data[z * n + k] = frac as f64;
                }
            }
            return Ok(Stack3 {
                planes: ncat,
                ny: self.nye,
                nx: self.nxe,
                data,
            });
        }

        if idx.kind != SourceType::Categorical {
            return Err(StaticError::Invalid(format!(
                "unsupported category source type {:?}",
                idx.kind
            )));
        }
        if idx.nz() != 1 {
            return Err(StaticError::Invalid(format!(
                "categorical source must have one z plane, got {}",
                idx.nz()
            )));
        }
        let flat = self.cells(ds, win)?;
        let mut counts = vec![0i64; n * ncat];
        for (&cell, &r) in flat.iter().zip(win.raw.iter()) {
            if cell >= 0 && r >= cmin && r <= cmax {
                counts[cell as usize * ncat + (r - cmin) as usize] += 1;
            }
        }
        // frac layout here: [cell][cat] (the Python's pre-moveaxis
        // (nye, nxe, ncat)); converted to plane-major on return.
        let mut frac = vec![0.0f64; n * ncat];
        let mut empty = vec![false; n];
        for cell in 0..n {
            let tot: i64 =
                counts[cell * ncat..(cell + 1) * ncat].iter().sum();
            let totf = tot as f64;
            empty[cell] = tot == 0;
            // WPS stores this working field as default REAL; its
            // optimized loop forms one float32 reciprocal per cell and
            // multiplies each category.
            let denom = totf.max(1.0) as f32;
            let recip = 1.0f32 / denom;
            for c in 0..ncat {
                frac[cell * ncat + c] =
                    ((counts[cell * ncat + c] as f32) * recip) as f64;
            }
        }
        if empty.iter().any(|&e| e) {
            let coords = self.cell_coords(ds, win);
            for cell in 0..n {
                if !empty[cell] {
                    continue;
                }
                let (mut ii, jj) = match &coords {
                    CoordArray::F32 { xi, yi } => (
                        ((xi[cell] + 0.5f32).floor() as i64) - win.x0,
                        ((yi[cell] + 0.5f32).floor() as i64) - win.y0,
                    ),
                    CoordArray::F64 { xi, yi } => (
                        ((xi[cell] + 0.5).floor() as i64) - win.x0,
                        ((yi[cell] + 0.5).floor() as i64) - win.y0,
                    ),
                };
                if ds.wraps_x && ii < 0 {
                    // cell_coords speaks DATASET coordinates; a window
                    // that crossed the wrap seam runs past nx_global,
                    // so re-frame in integer index space.  Doing it to
                    // the float coordinate instead is the lossy round
                    // trip that made two builds of the same ground
                    // disagree.
                    ii += ds.nx_global;
                }
                let inside = ii >= 0
                    && (ii as usize) < win.nx
                    && jj >= 0
                    && (jj as usize) < win.ny;
                if !inside {
                    continue;
                }
                let cat =
                    win.raw[jj as usize * win.nx + ii as usize];
                if cat >= cmin && cat <= cmax {
                    frac[cell * ncat + (cat - cmin) as usize] = 1.0;
                }
            }
        }
        let mut data = vec![0.0f64; ncat * n];
        for cell in 0..n {
            for c in 0..ncat {
                data[c * n + cell] = frac[cell * ncat + c];
            }
        }
        Ok(Stack3 {
            planes: ncat,
            ny: self.nye,
            nx: self.nxe,
            data,
        })
    }
}

// -- coverage receipt ---------------------------------------------------

#[derive(serde::Serialize)]
struct SourceGeometry {
    nx_global: i64,
    ny_global: i64,
    wraps_x: bool,
    extent_basis: String,
    tile_inventory_bounds: [i64; 4],
}

#[derive(serde::Serialize)]
struct SourceWindow {
    x_start: i64,
    x_end: i64,
    y_start: i64,
    y_end: i64,
}

#[derive(serde::Serialize)]
struct RequiredTile {
    origin: [i64; 2],
    relative_path: String,
    bytes: u64,
}

#[derive(serde::Serialize)]
struct CoverageReceipt {
    schema: &'static str,
    status: &'static str,
    field: String,
    dataset: String,
    declared_sparse: bool,
    source_geometry: SourceGeometry,
    source_window: SourceWindow,
    required_cells: usize,
    covered_cells: usize,
    missing_tile_cells: usize,
    outside_extent_cells: usize,
    coverage_fraction: f64,
    required_tile_count: usize,
    required_tiles: Vec<RequiredTile>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::{
        assert_bits_f32, assert_bits_f64, golden_dir, json, load_package,
        read_f64, read_i64,
    };

    #[test]
    fn pixel_binning_matches_the_python_including_knife_edges() {
        let dir = golden_dir().join("binning");
        let spec = json(&dir.join("goldens.json"));
        let (_, gx) = read_f64(&dir.join(spec["gx"].as_str().unwrap()));
        let (_, gy) = read_f64(&dir.join(spec["gy"].as_str().unwrap()));
        let (_, want) =
            read_i64(&dir.join(spec["flat"].as_str().unwrap()));
        let got = bin_grid_coords(
            &gx,
            &gy,
            spec["nxe"].as_u64().unwrap() as usize,
            spec["nye"].as_u64().unwrap() as usize,
            spec["halo"].as_u64().unwrap() as usize,
        );
        assert_eq!(got, want, "crafted knife-edge binning");
    }

    fn check_package_mesh(tag: &str) {
        let Some(pkg) = load_package(tag) else {
            eprintln!("SKIP: WPS_GEOG reference tree not present");
            return;
        };
        let mesh = pkg.mesh();
        let mo = &pkg.meta["mesh_out"];
        let (_, lat_e) = crate::testsupport::read_f32(&pkg.bin(&mo["lat_e"]));
        assert_bits_f32(&mesh.lat_e, &lat_e, &format!("{tag}: lat_e"));
        let (_, lat_lower) =
            crate::testsupport::read_f32(&pkg.bin(&mo["lat_lower"]));
        assert_bits_f32(
            &mesh.lat_lower_e,
            &lat_lower,
            &format!("{tag}: lat_lower"),
        );
        match (&mesh.lon_e, pkg.meta["lon_e_dtype"].as_str().unwrap()) {
            (LonE::F32(got), "f32") => {
                let (_, want) =
                    crate::testsupport::read_f32(&pkg.bin(&mo["lon_e"]));
                assert_bits_f32(got, &want, &format!("{tag}: lon_e"));
            }
            (LonE::F64(got), "f64") => {
                let (_, want) = read_f64(&pkg.bin(&mo["lon_e"]));
                assert_bits_f64(got, &want, &format!("{tag}: lon_e"));
            }
            (got, want) => panic!(
                "{tag}: lon_e dtype mismatch (got {}, want {want})",
                match got {
                    LonE::F32(_) => "f32",
                    LonE::F64(_) => "f64",
                }
            ),
        }
        for (name, got) in [
            ("lon_band", &mesh.lon_boundary_band),
            ("lat_band", &mesh.lat_integer_band),
        ] {
            let (_, want) =
                crate::testsupport::read_bool(&pkg.bin(&mo[name]));
            assert_eq!(got, &want, "{tag}: {name}");
        }
    }

    #[test]
    fn coarse_mesh_assembly_matches_the_python_sampler() {
        check_package_mesh("coarse");
    }

    #[test]
    fn subkm_mesh_assembly_matches_the_python_sampler() {
        check_package_mesh("subkm");
    }

    fn check_package_windows_and_coords(tag: &str) {
        let Some(pkg) = load_package(tag) else {
            eprintln!("SKIP: WPS_GEOG reference tree not present");
            return;
        };
        let dom = pkg.sampler();
        for (field, entry) in pkg.meta["windows"].as_object().unwrap() {
            let ds = GeogDataset::open(
                &pkg.geog.join(entry["dir"].as_str().unwrap()),
                None,
            )
            .unwrap();
            let win = dom.window(&ds, 3).unwrap_or_else(|err| {
                panic!("{tag}/{field}: window refused: {err}")
            });
            assert_eq!(
                win.x0,
                entry["x0"].as_i64().unwrap(),
                "{tag}/{field}: window x0"
            );
            assert_eq!(
                win.y0,
                entry["y0"].as_i64().unwrap(),
                "{tag}/{field}: window y0"
            );
            let shape: Vec<usize> = entry["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(
                vec![win.nz, win.ny, win.nx],
                shape,
                "{tag}/{field}: window shape"
            );
        }
        // cell_coords against the committed Python arrays.
        let probe = pkg.meta["cell_coords"]["field"].as_str().unwrap();
        let ds = GeogDataset::open(
            &pkg.geog
                .join(pkg.meta["windows"][probe]["dir"].as_str().unwrap()),
            None,
        )
        .unwrap();
        let win = dom.window(&ds, 3).unwrap();
        match dom.cell_coords(&ds, &win) {
            CoordArray::F64 { xi, yi } => {
                let (_, wx) =
                    read_f64(&pkg.bin(&pkg.meta["cell_coords"]["xi"]));
                let (_, wy) =
                    read_f64(&pkg.bin(&pkg.meta["cell_coords"]["yi"]));
                assert_bits_f64(&xi, &wx, &format!("{tag}: coords xi"));
                assert_bits_f64(&yi, &wy, &format!("{tag}: coords yi"));
            }
            CoordArray::F32 { xi, yi } => {
                let (_, wx) = crate::testsupport::read_f32(
                    &pkg.bin(&pkg.meta["cell_coords"]["xi"]),
                );
                let (_, wy) = crate::testsupport::read_f32(
                    &pkg.bin(&pkg.meta["cell_coords"]["yi"]),
                );
                assert_bits_f32(&xi, &wx, &format!("{tag}: coords xi"));
                assert_bits_f32(&yi, &wy, &format!("{tag}: coords yi"));
            }
        }
    }

    #[test]
    fn coarse_windows_and_cell_coords_match_the_python() {
        check_package_windows_and_coords("coarse");
    }

    #[test]
    fn subkm_windows_and_cell_coords_match_the_python() {
        check_package_windows_and_coords("subkm");
    }

    #[test]
    fn real_window_binning_reproduces_the_committed_cells() {
        let Some(pkg) = load_package("coarse") else {
            eprintln!("SKIP: WPS_GEOG reference tree not present");
            return;
        };
        let dom = pkg.sampler();
        let (_, gx) = read_f64(&pkg.bin(&pkg.meta["binning"]["gx"]));
        let (_, gy) = read_f64(&pkg.bin(&pkg.meta["binning"]["gy"]));
        let field = pkg.meta["binning"]["field"].as_str().unwrap();
        let (_, want) = read_i64(&pkg.dir.join(
            pkg.meta["windows"][field]["cells"].as_str().unwrap(),
        ));
        let got = bin_grid_coords(&gx, &gy, dom.nxe, dom.nye, dom.halo);
        assert_eq!(got, want, "terrain-window binning");
    }

    #[test]
    fn continuous_category_source_matches_the_python_normalization() {
        use std::collections::BTreeMap;
        use std::sync::Arc;
        let dir = golden_dir().join("contcat");
        let spec = json(&dir.join("goldens.json"));
        let ds = GeogDataset::open(
            &golden_dir().join("synthetic").join("syn_contcat"),
            None,
        )
        .unwrap();
        let a: Vec<i64> = spec["window_args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let win = ds.read_window(a[0], a[1], a[2], a[3]).unwrap();
        let nx = spec["nx"].as_u64().unwrap() as usize;
        let ny = spec["ny"].as_u64().unwrap() as usize;
        let halo = spec["halo"].as_u64().unwrap() as usize;
        let (nxe, nye) = (nx + 2 * halo, ny + 2 * halo);
        let (_, lat_e) = crate::testsupport::read_f32(
            &dir.join(spec["lat_e"].as_str().unwrap()),
        );
        let (_, lon_e) =
            read_f64(&dir.join(spec["lon_e"].as_str().unwrap()));
        let mesh = SamplerMesh {
            nxe,
            nye,
            lat_lower_e: lat_e.clone(),
            lat_e,
            lon_e: LonE::F64(lon_e),
            lon_boundary_band: vec![false; nxe * nye],
            lat_integer_band: vec![false; nxe * nye],
            lat_c: vec![0.0; (nxe + 1) * (nye + 1)],
            lon_c: vec![0.0; (nxe + 1) * (nye + 1)],
        };
        let hex = |key: &str| {
            f64::from_bits(
                u64::from_str_radix(spec[key].as_str().unwrap(), 16)
                    .unwrap(),
            )
        };
        // interp-only variant (ratio < 4 keeps gcell off)
        let dom = DomainSampler::from_parts(
            None,
            hex("dx_interp"),
            halo,
            nx,
            ny,
            mesh.clone(),
            BTreeMap::new(),
        )
        .unwrap();
        let got = dom.categorical(&ds, &win, false).unwrap();
        let (dims, want) =
            read_f64(&dir.join(spec["frac_interp"].as_str().unwrap()));
        assert_eq!(vec![got.planes, got.ny, got.nx], dims);
        assert_bits_f64(&got.data, &want, "contcat frac_interp");

        // gcell variant through the fixture-cells seam
        let key = crate::testsupport::Package::cells_key_public(
            &spec["key"],
        );
        let (_, flat) =
            read_i64(&dir.join(spec["cells"].as_str().unwrap()));
        let mut fixtures = BTreeMap::new();
        fixtures.insert(key, Arc::new(flat));
        let dom = DomainSampler::from_parts(
            None,
            hex("dx_gcell"),
            halo,
            nx,
            ny,
            mesh,
            fixtures,
        )
        .unwrap();
        let got = dom.categorical(&ds, &win, true).unwrap();
        let (_, want) =
            read_f64(&dir.join(spec["frac_gcell"].as_str().unwrap()));
        assert_bits_f64(&got.data, &want, "contcat frac_gcell");
    }

    #[test]
    fn coverage_receipt_matches_the_python_document() {
        let Some(pkg) = load_package("coarse") else {
            eprintln!("SKIP: WPS_GEOG reference tree not present");
            return;
        };
        let dom = pkg.sampler();
        let ds = GeogDataset::open(
            &pkg.geog.join(
                pkg.meta["windows"]["terrain"]["dir"].as_str().unwrap(),
            ),
            None,
        )
        .unwrap();
        let win = dom.window(&ds, 3).unwrap();
        let receipt = dom
            .require_source_coverage(&ds, &win, "terrain")
            .expect("coverage must pass on the reference tree");
        let got: serde_json::Value =
            serde_json::from_str(&receipt).unwrap();
        let mut want = json(&pkg.dir.join("receipt_terrain.json"));
        // The dataset path is machine-local; require agreement on the
        // directory name and compare the rest structurally.
        let got_ds = got["dataset"].as_str().unwrap();
        assert!(
            got_ds.ends_with(
                pkg.meta["windows"]["terrain"]["dir"].as_str().unwrap()
            ),
            "dataset path {got_ds:?} does not name the terrain dir"
        );
        want["dataset"] = got["dataset"].clone();
        assert_eq!(got, want, "terrain coverage receipt");
    }
}

/// `Path.resolve()`-style display: canonical absolute path without the
/// Windows verbatim prefix.
pub(crate) fn resolved_path_string(path: &Path) -> String {
    let resolved = std::fs::canonicalize(path)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| path.display().to_string());
    if let Some(rest) = resolved.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{rest}")
    } else if let Some(rest) = resolved.strip_prefix(r"\\?\") {
        rest.to_string()
    } else {
        resolved
    }
}
