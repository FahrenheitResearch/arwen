//! LANE 1.  Worldwide map-projection grids with WPS conventions.
//!
//! Port of `gpuwm/static/projection.py` + `lambert.py`: float64
//! transforms transcribed from WRF v4.6.1 `share/module_llxy.F`
//! (set_lc/ijll_lc/llij_lc, set_merc/…, set_ps/…), derived MAPFAC and
//! SINALPHA/COSALPHA from WPS geogrid `process_tile_module.F`
//! (get_map_factor, get_rotang), and the float32 WPS *sampling twins*
//! (`_WpsLambert32`/`_WpsMerc32`/`_WpsPs32`/`_TranslatedWps32` in
//! `build.py`) including their GNU-scalar ULP nudges and compiler-band
//! reconciliation inputs -- the twins select source stencils, so their
//! float32 arithmetic is part of the byte-parity contract.
//!
//! Parity: byte-identical to the Python float64/float32 results on the
//! committed golden domains (`tests/lane1_goldens.rs`, extracted by
//! `tools/static_rust_port/extract_lane1_goldens.py` from the real
//! Python).  The libm ledger lives in [`npmath`]: float64 and most
//! float32 transcendentals go through `std` (bit-equal to the UCRT
//! libm numpy uses, measured); float32 sin/cos/exp/log go through the
//! numpy-kernel ports in `npmath` (numpy routes those four through its
//! own SIMD kernels, measured unequal to libm).
//!
//! The trait signatures below are the shared floor: lane 2's sampler
//! consumes `ProjectedGrid` + `Wps32Twin` (+ [`wps32::SamplingSurface`]
//! for the precomputed ULP band data), lane 3 consumes `ProjectedGrid`
//! for footprints and CRS parameters.  Lanes 2 and 3 MUST NOT reshape
//! them.

pub mod lambert;
pub mod mercator;
pub mod npmath;
pub mod polar;
pub mod wps32;

pub use wps32::{DEG32, RAD32};

use rayon::prelude::*;

use crate::error::{Result, StaticError};
use crate::types::{Grid2, Stagger};
use crate::OMEGA_E;

pub(crate) const RAD_PER_DEG: f64 = std::f64::consts::PI / 180.0;
pub(crate) const DEG_PER_RAD: f64 = 180.0 / std::f64::consts::PI;

/// `_wrap180`: wrap a longitude difference into (-180, 180], applying
/// each numpy `where` once (module_llxy cut-zone convention).
pub(crate) fn wrap180(d: f64) -> f64 {
    let d = if d > 180.0 { d - 360.0 } else { d };
    if d < -180.0 { d + 360.0 } else { d }
}

/// WPS map_proj selector.  WRF header codes: lambert=1, polar=2,
/// mercator=3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ProjectionKind {
    Lambert,
    Mercator,
    Polar,
}

impl ProjectionKind {
    pub fn wrf_code(self) -> i32 {
        match self {
            ProjectionKind::Lambert => 1,
            ProjectionKind::Polar => 2,
            ProjectionKind::Mercator => 3,
        }
    }

    fn python_class_name(self) -> &'static str {
        match self {
            ProjectionKind::Lambert => "LambertGrid",
            ProjectionKind::Mercator => "MercatorGrid",
            ProjectionKind::Polar => "PolarStereoGrid",
        }
    }
}

/// Constructor parameters, uniform across projections exactly as the
/// Python `ProjectedGrid.__init__` is (unused parameters validated and
/// carried for receipts).  `known_x`/`known_y` default to the mass-grid
/// centre `(e_we/2, e_sn/2)` when absent -- resolved by the caller
/// before this struct is built so the struct itself is total.
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct GridSpec {
    pub kind: ProjectionKind,
    pub ref_lat: f64,
    pub ref_lon: f64,
    pub truelat1: f64,
    pub truelat2: f64,
    pub stand_lon: f64,
    pub dx: f64,
    pub dy: f64,
    pub e_we: i64,
    pub e_sn: i64,
    pub known_x: f64,
    pub known_y: f64,
    pub moad_cen_lat: f64,
    pub moad_cen_lon: f64,
}

/// One projected WRF domain: spec + derived projection state + the
/// placement-translation bookkeeping (`ProjectedGrid.translated` in the
/// Python: a translated grid DELEGATES its transforms to the reference
/// grid at an exact integer index offset, so shared cells evaluate
/// through identical float arithmetic -- the bitwise statics-on-move
/// invariant).
#[derive(Debug, Clone)]
pub struct ProjectedGrid {
    pub spec: GridSpec,
    /// `-1.0` south, `+1.0` north (module_llxy map_set).
    pub hemi: f64,
    pub cen_lat: f64,
    pub cen_lon: f64,
    /// `Some((reference, (di, dj)))` on a translated grid; transforms
    /// delegate as `reference.ij_to_latlon(x + di, y + dj)`.
    pub translation: Option<(Box<ProjectedGrid>, (i64, i64))>,
    state: State,
}

/// Per-projection derived state (set_lc / set_merc / set_ps outputs).
#[derive(Debug, Clone)]
pub(crate) enum State {
    Lambert(lambert::LambertState),
    Mercator(mercator::MercatorState),
    Polar(polar::PolarState),
}

impl ProjectedGrid {
    /// Build a grid from its spec (set_* transcriptions).
    pub fn new(spec: GridSpec) -> Result<Self> {
        if (spec.dx - spec.dy).abs() > 1e-9 * spec.dx.abs() {
            return Err(StaticError::Invalid(format!(
                "{} requires dx == dy, got {} != {}",
                spec.kind.python_class_name(),
                spec.dx,
                spec.dy
            )));
        }
        // map_set (module_llxy.F:533-537): hemisphere from truelat1 for
        // every dx-carrying projection.
        let hemi = if spec.truelat1 < 0.0 { -1.0 } else { 1.0 };
        let state = match spec.kind {
            ProjectionKind::Lambert => State::Lambert(lambert::setup(&spec, hemi)),
            ProjectionKind::Mercator => State::Mercator(mercator::setup(&spec)),
            ProjectionKind::Polar => State::Polar(polar::setup(&spec, hemi)),
        };
        let mut grid = ProjectedGrid {
            spec,
            hemi,
            cen_lat: 0.0,
            cen_lon: 0.0,
            translation: None,
            state,
        };
        // The seam always carries a resolved known point, which is the
        // Python explicit-known_x path: the centre comes out of the
        // grid's own round trip.  (The Python centred-default shortcut
        // that copies ref_lat/ref_lon verbatim is metadata the Python
        // wrapper keeps computing on its side.)
        let (cen_lat, cen_lon) = grid.ij_to_latlon(
            grid.spec.e_we as f64 / 2.0,
            grid.spec.e_sn as f64 / 2.0,
        );
        grid.cen_lat = cen_lat;
        grid.cen_lon = cen_lon;
        Ok(grid)
    }

    pub(crate) fn state(&self) -> &State {
        &self.state
    }

    /// The set_* derived scalars, named as the Python attributes hold
    /// them (receipts and the golden tests pin these bits).
    pub fn state_scalars(&self) -> Vec<(&'static str, f64)> {
        match &self.state {
            State::Lambert(s) => vec![
                ("cone", s.cone),
                ("rebydx", s.rebydx),
                ("rsw", s.rsw),
                ("polei", s.polei),
                ("polej", s.polej),
            ],
            State::Mercator(s) => vec![("dlon", s.dlon), ("rsw", s.rsw)],
            State::Polar(s) => vec![
                ("rebydx", s.rebydx),
                ("rsw", s.rsw),
                ("polei", s.polei),
                ("polej", s.polej),
            ],
        }
    }

    /// Projection coordinate -> (lat, lon) degrees, float64.
    pub fn ij_to_latlon(&self, x: f64, y: f64) -> (f64, f64) {
        if let Some((reference, (di, dj))) = &self.translation {
            return reference.ij_to_latlon(x + *di as f64, y + *dj as f64);
        }
        match &self.state {
            State::Lambert(s) => {
                lambert::ij_to_latlon(s, &self.spec, self.hemi, x, y)
            }
            State::Mercator(s) => mercator::ij_to_latlon(s, &self.spec, x, y),
            State::Polar(s) => {
                polar::ij_to_latlon(s, &self.spec, self.hemi, x, y)
            }
        }
    }

    /// (lat, lon) degrees -> projection coordinate, float64.
    pub fn latlon_to_ij(&self, lat: f64, lon: f64) -> (f64, f64) {
        if let Some((reference, (di, dj))) = &self.translation {
            let (x, y) = reference.latlon_to_ij(lat, lon);
            return (x - *di as f64, y - *dj as f64);
        }
        match &self.state {
            State::Lambert(s) => {
                lambert::latlon_to_ij(s, &self.spec, self.hemi, lat, lon)
            }
            State::Mercator(s) => mercator::latlon_to_ij(s, &self.spec, lat, lon),
            State::Polar(s) => {
                polar::latlon_to_ij(s, &self.spec, self.hemi, lat, lon)
            }
        }
    }

    /// Map scale factor at one latitude (get_map_factor).
    ///
    /// Uses the grid's own state (a translated grid's cone is bit-equal
    /// to its reference's: `lc_cone` is a pure function of the shared
    /// true latitudes), exactly as the Python calls the instance method.
    pub fn map_factor(&self, lat: f64) -> f64 {
        match &self.state {
            State::Lambert(s) => lambert::map_factor(s, &self.spec, self.hemi, lat),
            State::Mercator(_) => mercator::map_factor(&self.spec, lat),
            State::Polar(_) => polar::map_factor(&self.spec, lat),
        }
    }

    /// (SINALPHA, COSALPHA) at one longitude (get_rotang).
    pub fn rotation(&self, lon: f64) -> (f64, f64) {
        match &self.state {
            State::Lambert(s) => lambert::rotation(s, &self.spec, lon),
            State::Mercator(_) => mercator::rotation(),
            State::Polar(_) => polar::rotation(&self.spec, lon),
        }
    }

    /// Staggered-mesh dims `(ny, nx)` and offsets `(xoff, yoff)`, WPS
    /// registration (mass (i, j); U (i-0.5, j); V (i, j-0.5); corner
    /// (i-0.5, j-0.5)).
    pub fn stagger_layout(&self, stagger: Stagger) -> (usize, usize, f64, f64) {
        let e_we = self.spec.e_we as usize;
        let e_sn = self.spec.e_sn as usize;
        match stagger {
            Stagger::Mass => (e_sn - 1, e_we - 1, 0.0, 0.0),
            Stagger::U => (e_sn - 1, e_we, -0.5, 0.0),
            Stagger::V => (e_sn, e_we - 1, 0.0, -0.5),
            Stagger::Corner => (e_sn, e_we, -0.5, -0.5),
        }
    }

    /// Staggered lat/lon arrays, `(lat, lon)` each row-major `(ny, nx)`
    /// per the Python `latlon_mass/u/v/c` (rayon-parallel over rows;
    /// per-cell arithmetic, so the parallel result is bit-equal to the
    /// serial one).
    pub fn latlon(&self, stagger: Stagger) -> (Grid2, Grid2) {
        let (ny, nx, xoff, yoff) = self.stagger_layout(stagger);
        let mut lat = Grid2::filled(ny, nx, 0.0);
        let mut lon = Grid2::filled(ny, nx, 0.0);
        lat.data
            .par_chunks_mut(nx)
            .zip(lon.data.par_chunks_mut(nx))
            .enumerate()
            .for_each(|(j, (lat_row, lon_row))| {
                let y = (j + 1) as f64 + yoff;
                for i in 0..nx {
                    let x = (i + 1) as f64 + xoff;
                    let (la, lo) = self.ij_to_latlon(x, y);
                    lat_row[i] = la;
                    lon_row[i] = lo;
                }
            });
        (lat, lon)
    }

    /// MAPFAC_* at one stagger (map factor over that stagger's lat).
    pub fn map_factor_array(&self, stagger: Stagger) -> Grid2 {
        let (lat, _) = self.latlon(stagger);
        let mut out = Grid2::filled(lat.ny, lat.nx, 0.0);
        out.data
            .par_iter_mut()
            .zip(lat.data.par_iter())
            .for_each(|(o, &la)| *o = self.map_factor(la));
        out
    }

    /// (F, E) = (2*Omega*sin(lat), 2*Omega*cos(lat)) at one stagger.
    pub fn coriolis_arrays(&self, stagger: Stagger) -> (Grid2, Grid2) {
        let (lat, _) = self.latlon(stagger);
        let mut f = Grid2::filled(lat.ny, lat.nx, 0.0);
        let mut e = Grid2::filled(lat.ny, lat.nx, 0.0);
        f.data
            .par_iter_mut()
            .zip(e.data.par_iter_mut())
            .zip(lat.data.par_iter())
            .for_each(|((fv, ev), &la)| {
                *fv = 2.0 * OMEGA_E * (la * RAD_PER_DEG).sin();
                *ev = 2.0 * OMEGA_E * (la * RAD_PER_DEG).cos();
            });
        (f, e)
    }

    /// (SINALPHA, COSALPHA) at one stagger.
    pub fn rotation_arrays(&self, stagger: Stagger) -> (Grid2, Grid2) {
        let (_, lon) = self.latlon(stagger);
        let mut sin = Grid2::filled(lon.ny, lon.nx, 0.0);
        let mut cos = Grid2::filled(lon.ny, lon.nx, 0.0);
        sin.data
            .par_iter_mut()
            .zip(cos.data.par_iter_mut())
            .zip(lon.data.par_iter())
            .for_each(|((sv, cv), &lo)| {
                let (s, c) = self.rotation(lo);
                *sv = s;
                *cv = c;
            });
        (sin, cos)
    }

    /// WPS parent_start/ratio nest arithmetic (`ProjectedGrid.nest`):
    /// nest mass coordinate x sits at parent mass coordinate
    /// `(i_parent_start - 0.5) + (x - 0.5)/ratio`; the child is a full
    /// grid whose known point is its own (1, 1) mass point.
    pub fn nest(
        &self,
        i_parent_start: i64,
        j_parent_start: i64,
        parent_grid_ratio: i64,
        e_we: i64,
        e_sn: i64,
        resolved_dx: Option<f64>,
        resolved_dy: Option<f64>,
    ) -> Result<ProjectedGrid> {
        let r = parent_grid_ratio;
        if r < 1 {
            return Err(StaticError::Invalid(format!(
                "parent_grid_ratio must be >= 1, got {r}"
            )));
        }
        let xp = (i_parent_start as f64 - 0.5) + 0.5 / r as f64;
        let yp = (j_parent_start as f64 - 0.5) + 0.5 / r as f64;
        let (lat11, lon11) = self.ij_to_latlon(xp, yp);
        let child_dx = resolved_dx.unwrap_or(self.spec.dx / r as f64);
        let child_dy = resolved_dy.unwrap_or(self.spec.dy / r as f64);
        ProjectedGrid::new(GridSpec {
            kind: self.spec.kind,
            ref_lat: lat11,
            ref_lon: lon11,
            truelat1: self.spec.truelat1,
            truelat2: self.spec.truelat2,
            stand_lon: self.spec.stand_lon,
            dx: child_dx,
            dy: child_dy,
            e_we,
            e_sn,
            known_x: 1.0,
            known_y: 1.0,
            moad_cen_lat: self.spec.moad_cen_lat,
            moad_cen_lon: self.spec.moad_cen_lon,
        })
    }

    /// Whole-cell placement translation with optional re-extent
    /// (`ProjectedGrid.translated`); composes offsets onto the ORIGINAL
    /// reference so a long move chain never accumulates float error.
    pub fn translated(
        &self,
        di_cells: i64,
        dj_cells: i64,
        e_we: Option<i64>,
        e_sn: Option<i64>,
    ) -> Result<ProjectedGrid> {
        let (base, di_total, dj_total) = match &self.translation {
            None => (self, di_cells, dj_cells),
            Some((reference, (di0, dj0))) => {
                (reference.as_ref(), di0 + di_cells, dj0 + dj_cells)
            }
        };
        let e_we = e_we.unwrap_or(self.spec.e_we);
        let e_sn = e_sn.unwrap_or(self.spec.e_sn);
        if e_we < 2 || e_sn < 2 {
            return Err(StaticError::Invalid(format!(
                "a translated extent needs at least one cell per axis, \
                 got e_we={e_we}, e_sn={e_sn}"
            )));
        }
        let mut spec = base.spec.clone();
        spec.known_x = base.spec.known_x - di_total as f64;
        spec.known_y = base.spec.known_y - dj_total as f64;
        spec.e_we = e_we;
        spec.e_sn = e_sn;
        let mut new = ProjectedGrid::new(spec)?;
        new.translation = Some((Box::new(base.clone()), (di_total, dj_total)));
        // Metadata recomputed through the delegated transform, so the
        // moved footprint's centre is exact reference arithmetic too.
        let (cen_lat, cen_lon) = new.ij_to_latlon(
            new.spec.e_we as f64 / 2.0,
            new.spec.e_sn as f64 / 2.0,
        );
        new.cen_lat = cen_lat;
        new.cen_lon = cen_lon;
        Ok(new)
    }
}

/// The float32 WPS sampling twin: single-precision transforms used
/// while SELECTING source cells (stencil selection is control flow, so
/// this arithmetic is part of the byte-parity contract).  Lane 1 owns
/// every implementation, including the translated-delegation twin and
/// `adopt_public_pole` for sub-kilometre nests.
pub trait Wps32Twin {
    fn ij_to_latlon32(&self, x: f32, y: f32) -> (f32, f32);
    fn latlon_to_ij32(&self, lat: f32, lon: f32) -> (f32, f32);
    /// Sub-kilometre nests take the float64 grid's pole solution
    /// (WPS locates nests from their mass-grid centre).
    fn adopt_public_pole(&mut self, grid: &ProjectedGrid);
}

/// Build the sampling twin for one grid (`_wps32_for`).
pub fn wps32_for(grid: &ProjectedGrid) -> Result<Box<dyn Wps32Twin>> {
    wps32::twin_for(grid)
}
