//! LANE 1.  The float32 WPS sampling twins.
//!
//! Ports `_WpsLambert32`, `_WpsMerc32`, `_WpsPs32` and
//! `_TranslatedWps32` from `gpuwm/static/build.py` EXACTLY, including:
//!
//! * every intermediate held in f32 (WPS declares the geogrid map state
//!   default REAL);
//! * the Lambert twin's double `nextafter` poleward nudge in
//!   `ij_to_latlon` (GNU Fortran's scalar libm lands 1-2 float32 ULPs
//!   poleward of numpy's vector ufunc sequence -- the nudge preserves
//!   stencil selection at exact source-grid rows);
//! * NO nudges on Mercator/polar (the band reconciliations were
//!   measured against GNU geogrid output for Lambert only; documented
//!   in the Python docstrings);
//! * the translated twin's index-offset delegation (never rebuild the
//!   pole from a shifted known point -- re-rounding in a different
//!   binade can flip knife-edge stencil selections on shared ground).
//!
//! float32 sin/cos/exp/log go through [`super::npmath`]'s numpy-kernel
//! ports; tan/atan/atan2/asin/acos/log10/sqrt/pow go through `std`
//! (bit-equal to numpy on the reference platform, measured -- see the
//! `npmath` module docs).
//!
//! The compiler-band data the sampler consumes (`_lon_boundary_band`,
//! `_lat_integer_band`, `_geogrid_longitude` band stepping) is computed
//! HERE from the twin + public transforms, exposed as plain masks via
//! [`SamplingSurface`], so lane 2 consumes booleans and never
//! re-derives float32 ULP logic.

use super::npmath::{
    nextafter_down, nextafter_up, np_cosf, np_expf, np_logf, np_modf,
    np_powf, np_sinf, spacing_abs,
};
use super::{ProjectedGrid, ProjectionKind, State, Wps32Twin};
use crate::error::Result;
use crate::EARTH_RADIUS_M;

/// `f(np.pi / 180.0)` -- the twins' single-precision degree/radian
/// constants (public: the golden tests pin their bits).
pub const RAD32: f32 = (std::f64::consts::PI / 180.0) as f32;
pub const DEG32: f32 = (180.0 / std::f64::consts::PI) as f32;

/// `_WpsLambert32`.  Fields mirror the Python attributes (public so the
/// golden tests can pin every state bit).
pub struct LambertTwin {
    pub hemi: f32,
    pub tl1: f32,
    pub tl2: f32,
    pub cone: f32,
    pub rebydx: f32,
    pub stand_lon: f32,
    pub rsw: f32,
    pub polei: f32,
    pub polej: f32,
}

impl LambertTwin {
    pub fn new(grid: &ProjectedGrid) -> Self {
        let spec = &grid.spec;
        let hemi: f32 = if spec.truelat1 < 0.0 { -1.0 } else { 1.0 };
        let tl1 = spec.truelat1 as f32;
        let tl2 = spec.truelat2 as f32;
        let cone: f32 = if (spec.truelat1 - spec.truelat2).abs() > 0.1 {
            let num = np_cosf(tl1 * RAD32).log10() - np_cosf(tl2 * RAD32).log10();
            let den = ((45.0f32 - tl1.abs() / 2.0) * RAD32).tan().log10()
                - ((45.0f32 - tl2.abs() / 2.0) * RAD32).tan().log10();
            num / den
        } else {
            np_sinf(tl1.abs() * RAD32)
        };
        let rebydx = EARTH_RADIUS_M as f32 / spec.dx as f32;
        let stand_lon = spec.stand_lon as f32;
        let mut dlon = spec.ref_lon as f32 - stand_lon;
        if dlon > 180.0 {
            dlon -= 360.0;
        }
        if dlon < -180.0 {
            dlon += 360.0;
        }
        let ctl1r = np_cosf(tl1 * RAD32);
        let rsw = rebydx * ctl1r / cone
            * np_powf(
                ((90.0f32 * hemi - spec.ref_lat as f32) * RAD32 / 2.0).tan()
                    / ((90.0f32 * hemi - tl1) * RAD32 / 2.0).tan(),
                cone,
            );
        let arg = cone * dlon * RAD32;
        let polei = hemi * spec.known_x as f32 - hemi * rsw * np_sinf(arg);
        let polej = hemi * spec.known_y as f32 + rsw * np_cosf(arg);
        LambertTwin { hemi, tl1, tl2, cone, rebydx, stand_lon, rsw, polei, polej }
    }
}

impl Wps32Twin for LambertTwin {
    fn ij_to_latlon32(&self, x: f32, y: f32) -> (f32, f32) {
        let xx = self.hemi * x - self.polei;
        let yy = self.polej - self.hemi * y;
        let r2 = xx * xx + yy * yy;
        let r = r2.sqrt() / self.rebydx;
        let mut lon =
            self.stand_lon + DEG32 * (self.hemi * xx).atan2(yy) / self.cone;
        let chi1 = (90.0f32 - self.hemi * self.tl1) * RAD32;
        let chi2 = (90.0f32 - self.hemi * self.tl2) * RAD32;
        let chi = if chi1 == chi2 {
            2.0f32
                * (np_powf(r / chi1.tan(), 1.0f32 / self.cone)
                    * (chi1 * 0.5).tan())
                .atan()
        } else {
            2.0f32
                * (np_powf(r * self.cone / np_sinf(chi1), 1.0f32 / self.cone)
                    * (chi1 * 0.5).tan())
                .atan()
        };
        let mut lat = (90.0f32 - chi * DEG32) * self.hemi;
        if r2 == 0.0 {
            lat = 90.0 * self.hemi;
        }
        // GNU Fortran's scalar evaluation of the WPS expression lands
        // one to two float32 ULPs poleward of NumPy's vector ufunc
        // sequence.  Preserve that stencil-selecting result.
        lat = if self.hemi > 0.0 {
            nextafter_up(nextafter_up(lat))
        } else {
            nextafter_down(nextafter_down(lat))
        };
        lon = np_modf(lon + 360.0, 360.0);
        if lon > 180.0 {
            lon -= 360.0;
        }
        (lat, lon)
    }

    fn latlon_to_ij32(&self, lat: f32, lon: f32) -> (f32, f32) {
        let mut dlon = lon - self.stand_lon;
        if dlon > 180.0 {
            dlon -= 360.0;
        }
        if dlon < -180.0 {
            dlon += 360.0;
        }
        let ctl1r = np_cosf(self.tl1 * RAD32);
        let rm = self.rebydx * ctl1r / self.cone
            * np_powf(
                ((90.0f32 * self.hemi - lat) * RAD32 / 2.0).tan()
                    / ((90.0f32 * self.hemi - self.tl1) * RAD32 / 2.0).tan(),
                self.cone,
            );
        let arg = self.cone * dlon * RAD32;
        let x = self.polei + self.hemi * rm * np_sinf(arg);
        let y = self.polej - rm * np_cosf(arg);
        (self.hemi * x, self.hemi * y)
    }

    fn adopt_public_pole(&mut self, grid: &ProjectedGrid) {
        if let State::Lambert(state) = grid.state() {
            self.polei = state.polei as f32;
            self.polej = state.polej as f32;
            self.rebydx = state.rebydx as f32;
        }
    }
}

/// `_WpsMerc32`.
pub struct MercatorTwin {
    pub lat1: f32,
    pub lon1: f32,
    pub knowni: f32,
    pub knownj: f32,
    pub dlon: f32,
    pub rsw: f32,
}

impl MercatorTwin {
    pub fn new(grid: &ProjectedGrid) -> Self {
        let spec = &grid.spec;
        let lat1 = spec.ref_lat as f32;
        let lon1 = spec.ref_lon as f32;
        let knowni = spec.known_x as f32;
        let knownj = spec.known_y as f32;
        let clain = np_cosf(RAD32 * spec.truelat1 as f32);
        let dlon = spec.dx as f32 / (EARTH_RADIUS_M as f32 * clain);
        let mut rsw = 0.0f32;
        if spec.ref_lat != 0.0 {
            rsw = np_logf((0.5f32 * ((lat1 + 90.0) * RAD32)).tan()) / dlon;
        }
        MercatorTwin { lat1, lon1, knowni, knownj, dlon, rsw }
    }
}

impl Wps32Twin for MercatorTwin {
    fn ij_to_latlon32(&self, x: f32, y: f32) -> (f32, f32) {
        let lat = 2.0f32
            * np_expf(self.dlon * (self.rsw + y - self.knownj)).atan()
            * DEG32
            - 90.0;
        let mut lon = (x - self.knowni) * self.dlon * DEG32 + self.lon1;
        if lon > 180.0 {
            lon -= 360.0;
        }
        if lon < -180.0 {
            lon += 360.0;
        }
        (lat, lon)
    }

    fn latlon_to_ij32(&self, lat: f32, lon: f32) -> (f32, f32) {
        let mut dlon = lon - self.lon1;
        if dlon < -180.0 {
            dlon += 360.0;
        }
        if dlon > 180.0 {
            dlon -= 360.0;
        }
        let i = self.knowni + (dlon / (self.dlon * DEG32));
        let j = self.knownj
            + np_logf((0.5f32 * ((lat + 90.0) * RAD32)).tan()) / self.dlon
            - self.rsw;
        (i, j)
    }

    fn adopt_public_pole(&mut self, grid: &ProjectedGrid) {
        if let State::Mercator(state) = grid.state() {
            self.dlon = state.dlon as f32;
            self.rsw = state.rsw as f32;
        }
    }
}

/// `_WpsPs32`.
pub struct PolarTwin {
    pub hemi: f32,
    pub tl1: f32,
    pub stand_lon: f32,
    pub rebydx: f32,
    pub scale_top: f32,
    pub rsw: f32,
    pub polei: f32,
    pub polej: f32,
}

impl PolarTwin {
    pub fn new(grid: &ProjectedGrid) -> Self {
        let spec = &grid.spec;
        let hemi: f32 = if spec.truelat1 < 0.0 { -1.0 } else { 1.0 };
        let tl1 = spec.truelat1 as f32;
        let stand_lon = spec.stand_lon as f32;
        let rebydx = EARTH_RADIUS_M as f32 / spec.dx as f32;
        let reflon = stand_lon + 90.0;
        let scale_top = 1.0f32 + hemi * np_sinf(tl1 * RAD32);
        let ala1 = spec.ref_lat as f32 * RAD32;
        let rsw = rebydx * np_cosf(ala1) * scale_top
            / (1.0f32 + hemi * np_sinf(ala1));
        let alo1 = (spec.ref_lon as f32 - reflon) * RAD32;
        let polei = spec.known_x as f32 - rsw * np_cosf(alo1);
        let polej = spec.known_y as f32 - hemi * rsw * np_sinf(alo1);
        PolarTwin { hemi, tl1, stand_lon, rebydx, scale_top, rsw, polei, polej }
    }
}

impl Wps32Twin for PolarTwin {
    fn ij_to_latlon32(&self, x: f32, y: f32) -> (f32, f32) {
        let reflon = self.stand_lon + 90.0;
        let xx = x - self.polei;
        let yy = (y - self.polej) * self.hemi;
        let r2 = xx * xx + yy * yy;
        let gi2 = np_powf(self.rebydx * self.scale_top, 2.0);
        let mut lat = DEG32 * self.hemi * ((gi2 - r2) / (gi2 + r2)).asin();
        let arccos = (xx / r2.sqrt()).acos();
        let mut lon = if yy > 0.0 {
            reflon + DEG32 * arccos
        } else {
            reflon - DEG32 * arccos
        };
        if r2 == 0.0 {
            lat = self.hemi * 90.0;
            lon = reflon;
        }
        if lon > 180.0 {
            lon -= 360.0;
        }
        if lon < -180.0 {
            lon += 360.0;
        }
        (lat, lon)
    }

    fn latlon_to_ij32(&self, lat: f32, lon: f32) -> (f32, f32) {
        let reflon = self.stand_lon + 90.0;
        let ala = lat * RAD32;
        let rm = self.rebydx * np_cosf(ala) * self.scale_top
            / (1.0f32 + self.hemi * np_sinf(ala));
        let alo = (lon - reflon) * RAD32;
        let i = self.polei + rm * np_cosf(alo);
        let j = self.polej + self.hemi * rm * np_sinf(alo);
        (i, j)
    }

    fn adopt_public_pole(&mut self, grid: &ProjectedGrid) {
        if let State::Polar(state) = grid.state() {
            self.polei = state.polei as f32;
            self.polej = state.polej as f32;
            self.rebydx = state.rebydx as f32;
        }
    }
}

/// `_TranslatedWps32`: index-offset delegation for a translated grid's
/// twin.  Rebuilding the twin from the shifted known point would
/// re-round the pole in a different binade and could flip knife-edge
/// stencil selections on shared ground.
pub struct TranslatedTwin {
    inner: Box<dyn Wps32Twin>,
    di: f64,
    dj: f64,
}

impl Wps32Twin for TranslatedTwin {
    fn ij_to_latlon32(&self, x: f32, y: f32) -> (f32, f32) {
        self.inner
            .ij_to_latlon32(x + self.di as f32, y + self.dj as f32)
    }

    fn latlon_to_ij32(&self, lat: f32, lon: f32) -> (f32, f32) {
        let (x, y) = self.inner.latlon_to_ij32(lat, lon);
        (x - self.di as f32, y - self.dj as f32)
    }

    fn adopt_public_pole(&mut self, grid: &ProjectedGrid) {
        match &grid.translation {
            Some((reference, _)) => self.inner.adopt_public_pole(reference),
            None => self.inner.adopt_public_pole(grid),
        }
    }
}

/// `_wps32_for`: the sampling twin for one projected grid.
pub fn twin_for(grid: &ProjectedGrid) -> Result<Box<dyn Wps32Twin>> {
    if let Some((reference, (di, dj))) = &grid.translation {
        return Ok(Box::new(TranslatedTwin {
            inner: twin_for(reference)?,
            di: *di as f64,
            dj: *dj as f64,
        }));
    }
    Ok(match grid.spec.kind {
        ProjectionKind::Lambert => Box::new(LambertTwin::new(grid)),
        ProjectionKind::Mercator => Box::new(MercatorTwin::new(grid)),
        ProjectionKind::Polar => Box::new(PolarTwin::new(grid)),
    })
}

// ---------------------------------------------------------------------------
// Sampling surface: the `_DomainSampler.__init__` coordinate + ULP band
// precomputation, exposed to lane 2 as plain arrays and boolean masks.
// ---------------------------------------------------------------------------

/// Everything `_DomainSampler` derives from the twin + public
/// transforms before any source window is read.  Row-major
/// `(nye, nxe)` on the halo-extended mass mesh; `lat_c`/`lon_c` are on
/// the `(nye+1, nxe+1)` cell-corner mesh.
pub struct SamplingSurface {
    pub nye: usize,
    pub nxe: usize,
    /// Twin latitude (with the Lambert sub-km one-ULP absorption
    /// applied), f32 exactly as the Python holds it.
    pub lat_e: Vec<f32>,
    /// `nextafter(lat_e, -inf)`.
    pub lat_lower_e: Vec<f32>,
    /// Sampling longitude.  Holds the public float64 longitude on
    /// coarse grids; on sub-kilometre Lambert grids it holds the
    /// `_geogrid_longitude` float32 reconciliation widened exactly to
    /// f64 (`lon_e_is_f32` = true; the Python keeps that case in f32
    /// and immediately casts, so the widened copy is byte-faithful).
    pub lon_e: Vec<f64>,
    pub lon_e_is_f32: bool,
    /// GNU compiler-band masks (Lambert only; empty=false elsewhere).
    pub lon_boundary_band: Vec<bool>,
    pub lat_integer_band: Vec<bool>,
    /// Cell-corner twin latitude (f32) and public longitude (f64) for
    /// window bounds.
    pub lat_c: Vec<f32>,
    pub lon_c: Vec<f64>,
}

/// Mirror GNU WPS's optimized single-precision longitude result
/// (`_DomainSampler._geogrid_longitude`): bracket the WPS expression
/// with the public float64 transform to pick the compiler's evaluation
/// band, then move east by 0, 4 or 8 float32 ULPs.
pub fn geogrid_longitude(lon32: &[f32], lon64: &[f64]) -> Vec<f32> {
    lon32
        .iter()
        .zip(lon64)
        .map(|(&c, &d)| {
            let p = d as f32;
            let ci = c.to_bits() as i32;
            let pi = p.to_bits() as i32;
            let ulp = spacing_abs(p) as f64;
            let frac = if ulp != 0.0 { (d - p as f64) / ulp } else { 0.0 };
            let band = ci.wrapping_sub(pi);
            let steps: u8 = if band <= -3 || (band == -2 && frac < -0.05) {
                0
            } else if band >= 3 || (band == 2 && frac >= -0.05) {
                8
            } else {
                4
            };
            let mut out = c;
            for _ in 0..steps {
                out = nextafter_up(out);
            }
            out
        })
        .collect()
}

/// Build the sampling surface for one grid (`_DomainSampler.__init__`
/// through the band masks; window/source math stays with lane 2).
pub fn sampling_surface(
    grid: &ProjectedGrid,
    halo: usize,
) -> Result<SamplingSurface> {
    let nx = grid.spec.e_we as usize - 1;
    let ny = grid.spec.e_sn as usize - 1;
    let nxe = nx + 2 * halo;
    let nye = ny + 2 * halo;
    let is_lambert = grid.spec.kind == ProjectionKind::Lambert;
    let sub_km = grid.spec.dx < 1000.0;

    let mut twin = twin_for(grid)?;
    if sub_km {
        // WPS locates nests from their mass-grid centre; the public
        // float64 projection already carries the resulting pole.
        twin.adopt_public_pole(grid);
    }

    let n = nye * nxe;
    let mut lat32 = vec![0.0f32; n];
    let mut lon32 = vec![0.0f32; n];
    let mut lat64 = vec![0.0f64; n];
    let mut lon64 = vec![0.0f64; n];
    for j in 0..nye {
        let yf = (1 - halo as i64 + j as i64) as f64;
        for i in 0..nxe {
            let xf = (1 - halo as i64 + i as i64) as f64;
            let (la32, lo32) = twin.ij_to_latlon32(xf as f32, yf as f32);
            let (la64, lo64) = grid.ij_to_latlon(xf, yf);
            let k = j * nxe + i;
            lat32[k] = la32;
            lon32[k] = lo32;
            lat64[k] = la64;
            lon64[k] = lo64;
        }
    }

    let mut lat_e = lat32;
    let mut lat_lower_e: Vec<f32> =
        lat_e.iter().map(|&v| nextafter_down(v)).collect();
    if is_lambert && sub_km {
        // One of the two scalar-libm ULPs documented in ij_to_latlon is
        // absorbed when geogrid initializes a nest from its centre.
        lat_e = lat_lower_e;
        lat_lower_e = lat_e.iter().map(|&v| nextafter_down(v)).collect();
    }

    let (lon_e, lon_e_is_f32) = if is_lambert && sub_km {
        let reconciled = geogrid_longitude(&lon32, &lon64);
        (reconciled.iter().map(|&v| v as f64).collect(), true)
    } else {
        (lon64.clone(), false)
    };

    let (lon_boundary_band, lat_integer_band) = if is_lambert {
        let lon_band = lon32
            .iter()
            .zip(&lon64)
            .map(|(&c, &d)| {
                let p = d as f32;
                let ulp = spacing_abs(p) as f64;
                let frac = if ulp != 0.0 { (d - p as f64) / ulp } else { 0.0 };
                let band =
                    (c.to_bits() as i32).wrapping_sub(p.to_bits() as i32);
                band.wrapping_abs() == 2 && (-0.15..=-0.05).contains(&frac)
            })
            .collect();
        let lat_band = lat_lower_e
            .iter()
            .zip(&lat64)
            .map(|(&lower, &d)| {
                let p = d as f32;
                let ulp = spacing_abs(p) as f64;
                let frac = if ulp != 0.0 { (d - p as f64) / ulp } else { 0.0 };
                let band = (lower.to_bits() as i32)
                    .wrapping_sub(p.to_bits() as i32);
                band == 4 && (0.38..=0.41).contains(&frac)
            })
            .collect();
        (lon_band, lat_band)
    } else {
        (vec![false; n], vec![false; n])
    };

    // Cell-corner mesh of the extended grid, for window bounds: twin
    // latitude, public longitude (exactly the Python's split).
    let ncx = nxe + 1;
    let ncy = nye + 1;
    let mut lat_c = vec![0.0f32; ncy * ncx];
    let mut lon_c = vec![0.0f64; ncy * ncx];
    for j in 0..ncy {
        let yf = 0.5 - halo as f64 + j as f64;
        for i in 0..ncx {
            let xf = 0.5 - halo as f64 + i as f64;
            let (la32, _) = twin.ij_to_latlon32(xf as f32, yf as f32);
            let (_, lo64) = grid.ij_to_latlon(xf, yf);
            let k = j * ncx + i;
            lat_c[k] = la32;
            lon_c[k] = lo64;
        }
    }

    Ok(SamplingSurface {
        nye,
        nxe,
        lat_e,
        lat_lower_e,
        lon_e,
        lon_e_is_f32,
        lon_boundary_band,
        lat_integer_band,
        lat_c,
        lon_c,
    })
}
