//! Projection math and metadata adapters for projected map rendering.

use crate::overlay::MapExtent;
use rustwx_core as core;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

const R_EARTH: f64 = 6_370_000.0;
const DEG2RAD: f64 = PI / 180.0;
const RAD2DEG: f64 = 180.0 / PI;
const GEOGRAPHIC_INFERENCE_MIN_LAT_SPAN_DEG: f64 = 100.0;
const GEOGRAPHIC_INFERENCE_MIN_LON_SPAN_DEG: f64 = 300.0;
const ROBINSON_X: [f64; 19] = [
    1.0000, 0.9986, 0.9954, 0.9900, 0.9822, 0.9730, 0.9600, 0.9427, 0.9216, 0.8962, 0.8679, 0.8350,
    0.7986, 0.7597, 0.7186, 0.6732, 0.6213, 0.5722, 0.5322,
];
const ROBINSON_Y: [f64; 19] = [
    0.0000, 0.0620, 0.1240, 0.1860, 0.2480, 0.3100, 0.3720, 0.4340, 0.4958, 0.5571, 0.6176, 0.6769,
    0.7346, 0.7903, 0.8435, 0.8936, 0.9394, 0.9761, 1.0000,
];

/// Lightweight, render-local projection metadata.
///
/// The shape mirrors `rustwx_core::GridProjection` so callers can hand native
/// grid metadata through without coupling the render crate directly to parser
/// internals. When projection metadata is unavailable, callers can still omit
/// this and let the projected-map builder fall back to an inferred Lambert
/// conformal setup.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProjectionSpec {
    Geographic,
    Robinson {
        central_meridian_deg: f64,
    },
    AlbersEqualArea {
        standard_parallel_1_deg: f64,
        standard_parallel_2_deg: f64,
        central_meridian_deg: f64,
        latitude_of_origin_deg: f64,
    },
    LambertConformal {
        standard_parallel_1_deg: f64,
        standard_parallel_2_deg: f64,
        central_meridian_deg: f64,
    },
    PolarStereographic {
        true_latitude_deg: f64,
        central_meridian_deg: f64,
        south_pole_on_projection_plane: bool,
    },
    Mercator {
        latitude_of_true_scale_deg: f64,
        central_meridian_deg: f64,
    },
    Other {
        template: u16,
    },
}

impl ProjectionSpec {
    pub fn is_projected(&self) -> bool {
        !matches!(self, Self::Geographic)
    }

    /// Approximate fallback for legacy callers that only have a lat/lon mesh.
    ///
    /// This intentionally picks a region-centered Lambert conformal setup for
    /// midlatitude weather domains. If upstream native projection metadata is
    /// available, callers should prefer passing it explicitly instead of using
    /// this inference path.
    pub fn infer_from_latlon_grid(lat_deg: &[f32], lon_deg: &[f32]) -> Option<Self> {
        let stats = LatLonStats::from_grid(lat_deg, lon_deg)?;
        let lat_span = (stats.max_lat - stats.min_lat).abs();
        if lat_span >= GEOGRAPHIC_INFERENCE_MIN_LAT_SPAN_DEG
            && stats.lon_span >= GEOGRAPHIC_INFERENCE_MIN_LON_SPAN_DEG
        {
            return Some(Self::Geographic);
        }
        let (sp1, sp2) = inferred_standard_parallels(stats.min_lat, stats.max_lat, lat_span);
        Some(Self::LambertConformal {
            standard_parallel_1_deg: sp1,
            standard_parallel_2_deg: sp2,
            central_meridian_deg: stats.center_lon,
        })
    }

    pub(crate) fn build_projector(
        &self,
        reference_latitude_deg: Option<f64>,
        reference_longitude_deg: Option<f64>,
        lat_deg: &[f32],
        lon_deg: &[f32],
    ) -> Result<ProjectionProjector, &'static str> {
        match *self {
            Self::Geographic => {
                let center_lon = reference_longitude_deg
                    .map(normalize_longitude_deg)
                    .or_else(|| circular_mean_longitude_deg(lon_deg))
                    .ok_or("projection requires at least one finite longitude")?;
                Ok(ProjectionProjector::Geographic(GeographicProjection {
                    central_meridian_deg: center_lon,
                }))
            }
            Self::Robinson {
                central_meridian_deg,
            } => Ok(ProjectionProjector::Robinson(RobinsonProjection::new(
                central_meridian_deg,
            ))),
            Self::AlbersEqualArea {
                standard_parallel_1_deg,
                standard_parallel_2_deg,
                central_meridian_deg,
                latitude_of_origin_deg,
            } => Ok(ProjectionProjector::AlbersEqualArea(
                AlbersEqualAreaProjection::new(
                    standard_parallel_1_deg,
                    standard_parallel_2_deg,
                    central_meridian_deg,
                    latitude_of_origin_deg,
                ),
            )),
            Self::LambertConformal {
                standard_parallel_1_deg,
                standard_parallel_2_deg,
                central_meridian_deg,
            } => {
                let ref_lat = reference_latitude_deg
                    .or_else(|| latitude_midpoint_deg(lat_deg))
                    .unwrap_or((standard_parallel_1_deg + standard_parallel_2_deg) / 2.0);
                Ok(ProjectionProjector::LambertConformal(
                    LambertConformal::new(
                        standard_parallel_1_deg,
                        standard_parallel_2_deg,
                        central_meridian_deg,
                        stabilize_reference_latitude(ref_lat),
                    ),
                ))
            }
            Self::PolarStereographic {
                true_latitude_deg,
                central_meridian_deg,
                south_pole_on_projection_plane,
            } => Ok(ProjectionProjector::PolarStereographic(
                PolarStereographic::new(
                    true_latitude_deg,
                    central_meridian_deg,
                    south_pole_on_projection_plane,
                ),
            )),
            Self::Mercator {
                latitude_of_true_scale_deg,
                central_meridian_deg,
            } => Ok(ProjectionProjector::Mercator(MercatorProjection::new(
                latitude_of_true_scale_deg,
                central_meridian_deg,
            ))),
            Self::Other { .. } => Err("projection template is not supported by rustwx-render"),
        }
    }
}

impl From<core::GridProjection> for ProjectionSpec {
    fn from(value: core::GridProjection) -> Self {
        match value {
            core::GridProjection::Geographic => Self::Geographic,
            core::GridProjection::LambertConformal {
                standard_parallel_1_deg,
                standard_parallel_2_deg,
                central_meridian_deg,
            } => Self::LambertConformal {
                standard_parallel_1_deg,
                standard_parallel_2_deg,
                central_meridian_deg,
            },
            core::GridProjection::PolarStereographic {
                true_latitude_deg,
                central_meridian_deg,
                south_pole_on_projection_plane,
            } => Self::PolarStereographic {
                true_latitude_deg,
                central_meridian_deg,
                south_pole_on_projection_plane,
            },
            core::GridProjection::Mercator {
                latitude_of_true_scale_deg,
                central_meridian_deg,
            } => Self::Mercator {
                latitude_of_true_scale_deg,
                central_meridian_deg,
            },
            core::GridProjection::Other { template } => Self::Other { template },
        }
    }
}

impl From<ProjectionSpec> for core::GridProjection {
    fn from(value: ProjectionSpec) -> Self {
        match value {
            ProjectionSpec::Geographic => Self::Geographic,
            ProjectionSpec::Robinson { .. } => Self::Geographic,
            ProjectionSpec::AlbersEqualArea { .. } => Self::Geographic,
            ProjectionSpec::LambertConformal {
                standard_parallel_1_deg,
                standard_parallel_2_deg,
                central_meridian_deg,
            } => Self::LambertConformal {
                standard_parallel_1_deg,
                standard_parallel_2_deg,
                central_meridian_deg,
            },
            ProjectionSpec::PolarStereographic {
                true_latitude_deg,
                central_meridian_deg,
                south_pole_on_projection_plane,
            } => Self::PolarStereographic {
                true_latitude_deg,
                central_meridian_deg,
                south_pole_on_projection_plane,
            },
            ProjectionSpec::Mercator {
                latitude_of_true_scale_deg,
                central_meridian_deg,
            } => Self::Mercator {
                latitude_of_true_scale_deg,
                central_meridian_deg,
            },
            ProjectionSpec::Other { template } => Self::Other { template },
        }
    }
}

/// Lambert conformal conic projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LambertConformal {
    n: f64,
    f: f64,
    rho0: f64,
    lambda0: f64,
    truelat1_deg: f64,
    truelat2_deg: f64,
    stand_lon_deg: f64,
    ref_lat_deg: f64,
}

impl LambertConformal {
    /// Create from degrees.
    pub fn new(truelat1_deg: f64, truelat2_deg: f64, stand_lon_deg: f64, ref_lat_deg: f64) -> Self {
        let phi1 = stabilize_reference_latitude(truelat1_deg) * DEG2RAD;
        let phi2 = stabilize_reference_latitude(truelat2_deg) * DEG2RAD;
        let phi0 = stabilize_reference_latitude(ref_lat_deg) * DEG2RAD;
        let lambda0 = stand_lon_deg * DEG2RAD;

        let mut n = if (truelat1_deg - truelat2_deg).abs() < 1e-10 {
            phi1.sin()
        } else {
            let num = (phi1.cos()).ln() - (phi2.cos()).ln();
            let den = ((PI / 4.0 + phi2 / 2.0).tan()).ln() - ((PI / 4.0 + phi1 / 2.0).tan()).ln();
            num / den
        };
        if n.abs() < 1e-8 {
            let fallback = if phi0.abs() >= 1e-8 {
                phi0
            } else if phi1.abs() >= 1e-8 {
                phi1
            } else {
                10.0 * DEG2RAD
            };
            n = fallback.sin();
        }

        let f = phi1.cos() * (PI / 4.0 + phi1 / 2.0).tan().powf(n) / n;
        let rho0 = R_EARTH * f / (PI / 4.0 + phi0 / 2.0).tan().powf(n);

        Self {
            n,
            f,
            rho0,
            lambda0,
            truelat1_deg,
            truelat2_deg,
            stand_lon_deg,
            ref_lat_deg,
        }
    }

    pub fn project(&self, lat: f64, lon: f64) -> (f64, f64) {
        let phi = stabilize_latitude(lat) * DEG2RAD;
        let delta_lon = normalize_longitude_deg(lon - self.stand_lon_deg) * DEG2RAD;

        let rho = R_EARTH * self.f / (PI / 4.0 + phi / 2.0).tan().powf(self.n);
        let theta = self.n * delta_lon;

        let x = rho * theta.sin();
        let y = self.rho0 - rho * theta.cos();
        (x, y)
    }

    fn unproject(self, x: f64, y: f64) -> Option<(f64, f64)> {
        let rho_abs = (x * x + (self.rho0 - y).powi(2)).sqrt();
        if !rho_abs.is_finite()
            || rho_abs <= 0.0
            || self.n.abs() < 1.0e-12
            || self.f.abs() < 1.0e-12
        {
            return None;
        }
        let rho_sign = self.n.signum();
        let rho = rho_abs * rho_sign;
        let theta = (x * rho_sign).atan2((self.rho0 - y) * rho_sign);
        let ratio = R_EARTH * self.f / rho;
        if ratio <= 0.0 || !ratio.is_finite() {
            return None;
        }
        let phi = 2.0 * ratio.powf(1.0 / self.n).atan() - PI / 2.0;
        let lon = self.stand_lon_deg + theta / self.n * RAD2DEG;
        Some((phi * RAD2DEG, normalize_longitude_deg(lon)))
    }

    pub fn spec(self) -> ProjectionSpec {
        ProjectionSpec::LambertConformal {
            standard_parallel_1_deg: self.truelat1_deg,
            standard_parallel_2_deg: self.truelat2_deg,
            central_meridian_deg: self.stand_lon_deg,
        }
    }

    pub fn reference_latitude_deg(self) -> f64 {
        self.ref_lat_deg
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GeographicProjection {
    central_meridian_deg: f64,
}

impl GeographicProjection {
    fn project(self, lat: f64, lon: f64) -> (f64, f64) {
        (
            normalize_longitude_deg(lon - self.central_meridian_deg),
            stabilize_latitude(lat),
        )
    }

    fn unproject(self, x: f64, y: f64) -> Option<(f64, f64)> {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        Some((
            stabilize_latitude(y),
            normalize_longitude_deg(x + self.central_meridian_deg),
        ))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RobinsonProjection {
    central_meridian_deg: f64,
}

impl RobinsonProjection {
    fn new(central_meridian_deg: f64) -> Self {
        Self {
            central_meridian_deg,
        }
    }

    fn project(self, lat: f64, lon: f64) -> (f64, f64) {
        let lat = stabilize_latitude(lat);
        let lon = normalize_longitude_deg(lon - self.central_meridian_deg);
        let abs_lat = lat.abs().min(90.0);
        let band = (abs_lat / 5.0).floor().min(17.0) as usize;
        let t = (abs_lat - (band as f64 * 5.0)) / 5.0;
        let interp = |table: &[f64; 19]| table[band] + (table[band + 1] - table[band]) * t;
        let x = R_EARTH * 0.8487 * interp(&ROBINSON_X) * lon * DEG2RAD;
        let y = R_EARTH * 1.3523 * interp(&ROBINSON_Y) * lat.signum();
        (x, y)
    }

    fn unproject(self, x: f64, y: f64) -> Option<(f64, f64)> {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let scaled_y = (y / (R_EARTH * 1.3523)).abs();
        if scaled_y > 1.0 + 1.0e-9 {
            return None;
        }
        let mut band = 17usize;
        for idx in 0..18 {
            if scaled_y <= ROBINSON_Y[idx + 1] + 1.0e-12 {
                band = idx;
                break;
            }
        }
        let y0 = ROBINSON_Y[band];
        let y1 = ROBINSON_Y[band + 1];
        let t = if (y1 - y0).abs() > 1.0e-12 {
            ((scaled_y - y0) / (y1 - y0)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let lat_abs = ((band as f64) + t) * 5.0;
        let x_scale = ROBINSON_X[band] + (ROBINSON_X[band + 1] - ROBINSON_X[band]) * t;
        if x_scale <= 0.0 {
            return None;
        }
        let lon_delta = x / (R_EARTH * 0.8487 * x_scale) * RAD2DEG;
        if lon_delta.abs() > 180.0 + 1.0e-6 {
            return None;
        }
        Some((
            lat_abs.copysign(y),
            normalize_longitude_deg(self.central_meridian_deg + lon_delta),
        ))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AlbersEqualAreaProjection {
    n: f64,
    c: f64,
    rho0: f64,
    central_meridian_deg: f64,
}

impl AlbersEqualAreaProjection {
    fn new(
        standard_parallel_1_deg: f64,
        standard_parallel_2_deg: f64,
        central_meridian_deg: f64,
        latitude_of_origin_deg: f64,
    ) -> Self {
        let phi1 = stabilize_reference_latitude(standard_parallel_1_deg) * DEG2RAD;
        let phi2 = stabilize_reference_latitude(standard_parallel_2_deg) * DEG2RAD;
        let phi0 = stabilize_latitude(latitude_of_origin_deg) * DEG2RAD;
        let mut n = 0.5 * (phi1.sin() + phi2.sin());
        if n.abs() < 1.0e-8 {
            n = phi1.sin();
        }
        if n.abs() < 1.0e-8 {
            n = (10.0 * DEG2RAD).sin();
        }
        let c = phi1.cos().powi(2) + 2.0 * n * phi1.sin();
        let rho0 = Self::rho(phi0, n, c);
        Self {
            n,
            c,
            rho0,
            central_meridian_deg,
        }
    }

    fn project(self, lat: f64, lon: f64) -> (f64, f64) {
        let phi = stabilize_latitude(lat) * DEG2RAD;
        let theta = self.n * normalize_longitude_deg(lon - self.central_meridian_deg) * DEG2RAD;
        let rho = Self::rho(phi, self.n, self.c);
        let x = rho * theta.sin();
        let y = self.rho0 - rho * theta.cos();
        (x, y)
    }

    fn unproject(self, x: f64, y: f64) -> Option<(f64, f64)> {
        let rho_abs = (x * x + (self.rho0 - y).powi(2)).sqrt();
        if !rho_abs.is_finite() || rho_abs <= 0.0 || self.n.abs() < 1.0e-12 {
            return None;
        }
        let rho_sign = self.n.signum();
        let rho = rho_abs * rho_sign;
        let theta = (x * rho_sign).atan2((self.rho0 - y) * rho_sign);
        let arg = (self.c - (rho * self.n / R_EARTH).powi(2)) / (2.0 * self.n);
        if !arg.is_finite() || !(-1.0..=1.0).contains(&arg) {
            return None;
        }
        let lat = arg.asin() * RAD2DEG;
        let lon = self.central_meridian_deg + theta / self.n * RAD2DEG;
        Some((lat, normalize_longitude_deg(lon)))
    }

    fn rho(phi: f64, n: f64, c: f64) -> f64 {
        R_EARTH * (c - 2.0 * n * phi.sin()).max(0.0).sqrt() / n
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PolarStereographic {
    central_meridian_deg: f64,
    south_pole_on_projection_plane: bool,
    k: f64,
}

impl PolarStereographic {
    fn new(
        true_latitude_deg: f64,
        central_meridian_deg: f64,
        south_pole_on_projection_plane: bool,
    ) -> Self {
        let lat_ts = stabilize_reference_latitude(true_latitude_deg) * DEG2RAD;
        Self {
            central_meridian_deg,
            south_pole_on_projection_plane,
            k: (1.0 + lat_ts.sin()) / 2.0,
        }
    }

    fn project(self, lat: f64, lon: f64) -> (f64, f64) {
        let phi = stabilize_latitude(lat) * DEG2RAD;
        let theta = normalize_longitude_deg(lon - self.central_meridian_deg) * DEG2RAD;
        if self.south_pole_on_projection_plane {
            let rho = 2.0 * R_EARTH * self.k * (PI / 4.0 + phi / 2.0).tan();
            (rho * theta.sin(), rho * theta.cos())
        } else {
            let rho = 2.0 * R_EARTH * self.k * (PI / 4.0 - phi / 2.0).tan();
            (rho * theta.sin(), -rho * theta.cos())
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MercatorProjection {
    central_meridian_deg: f64,
    scale: f64,
}

impl MercatorProjection {
    fn new(latitude_of_true_scale_deg: f64, central_meridian_deg: f64) -> Self {
        Self {
            central_meridian_deg,
            scale: (stabilize_reference_latitude(latitude_of_true_scale_deg) * DEG2RAD)
                .cos()
                .max(1.0e-6),
        }
    }

    fn project(self, lat: f64, lon: f64) -> (f64, f64) {
        let phi = stabilize_latitude(lat) * DEG2RAD;
        let lambda = normalize_longitude_deg(lon - self.central_meridian_deg) * DEG2RAD;
        let x = R_EARTH * self.scale * lambda;
        let y = R_EARTH * self.scale * ((PI / 4.0 + phi / 2.0).tan()).ln();
        (x, y)
    }

    fn unproject(self, x: f64, y: f64) -> Option<(f64, f64)> {
        if !x.is_finite() || !y.is_finite() || self.scale <= 0.0 {
            return None;
        }
        let lon = self.central_meridian_deg + x / (R_EARTH * self.scale) * RAD2DEG;
        let lat = (2.0 * (y / (R_EARTH * self.scale)).exp().atan() - PI / 2.0) * RAD2DEG;
        Some((stabilize_latitude(lat), normalize_longitude_deg(lon)))
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct InverseProjectionKernelParams {
    pub(crate) kind: i32,
    pub(crate) p0: f64,
    pub(crate) p1: f64,
    pub(crate) p2: f64,
    pub(crate) p3: f64,
    pub(crate) p4: f64,
    pub(crate) p5: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ProjectionProjector {
    Geographic(GeographicProjection),
    Robinson(RobinsonProjection),
    AlbersEqualArea(AlbersEqualAreaProjection),
    LambertConformal(LambertConformal),
    PolarStereographic(PolarStereographic),
    Mercator(MercatorProjection),
}

impl ProjectionProjector {
    pub(crate) fn project(self, lat: f64, lon: f64) -> (f64, f64) {
        match self {
            Self::Geographic(projector) => projector.project(lat, lon),
            Self::Robinson(projector) => projector.project(lat, lon),
            Self::AlbersEqualArea(projector) => projector.project(lat, lon),
            Self::LambertConformal(projector) => projector.project(lat, lon),
            Self::PolarStereographic(projector) => projector.project(lat, lon),
            Self::Mercator(projector) => projector.project(lat, lon),
        }
    }

    pub(crate) fn unproject(self, x: f64, y: f64) -> Option<(f64, f64)> {
        match self {
            Self::Geographic(projector) => projector.unproject(x, y),
            Self::AlbersEqualArea(projector) => projector.unproject(x, y),
            Self::LambertConformal(projector) => projector.unproject(x, y),
            Self::Mercator(projector) => projector.unproject(x, y),
            Self::Robinson(projector) => projector.unproject(x, y),
            Self::PolarStereographic(_) => None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn inverse_kernel_params(self) -> Option<InverseProjectionKernelParams> {
        let params = match self {
            // kind values are mirrored by kernels/render/rasterize_inverse_projected_grid.cu.
            Self::Geographic(projector) => InverseProjectionKernelParams {
                kind: 0,
                p0: projector.central_meridian_deg,
                p1: 0.0,
                p2: 0.0,
                p3: 0.0,
                p4: 0.0,
                p5: 0.0,
            },
            Self::Robinson(projector) => InverseProjectionKernelParams {
                kind: 1,
                p0: projector.central_meridian_deg,
                p1: 0.0,
                p2: 0.0,
                p3: 0.0,
                p4: 0.0,
                p5: 0.0,
            },
            Self::AlbersEqualArea(projector) => InverseProjectionKernelParams {
                kind: 2,
                p0: projector.n,
                p1: projector.c,
                p2: projector.rho0,
                p3: projector.central_meridian_deg,
                p4: 0.0,
                p5: 0.0,
            },
            Self::LambertConformal(projector) => InverseProjectionKernelParams {
                kind: 3,
                p0: projector.n,
                p1: projector.f,
                p2: projector.rho0,
                p3: projector.stand_lon_deg,
                p4: 0.0,
                p5: 0.0,
            },
            Self::Mercator(projector) => InverseProjectionKernelParams {
                kind: 4,
                p0: projector.central_meridian_deg,
                p1: projector.scale,
                p2: 0.0,
                p3: 0.0,
                p4: 0.0,
                p5: 0.0,
            },
            Self::PolarStereographic(_) => return None,
        };
        Some(params)
    }
}

impl MapExtent {
    pub fn from_wrf(
        proj: &LambertConformal,
        cen_lat: f64,
        cen_lon: f64,
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
    ) -> Self {
        let (xc, yc) = proj.project(cen_lat, cen_lon);
        Self {
            x_min: xc - dx * (nx as f64 - 1.0) / 2.0,
            x_max: xc + dx * (nx as f64 - 1.0) / 2.0,
            y_min: yc - dy * (ny as f64 - 1.0) / 2.0,
            y_max: yc + dy * (ny as f64 - 1.0) / 2.0,
        }
    }

    pub fn from_bounds(x_min: f64, x_max: f64, y_min: f64, y_max: f64, target_ratio: f64) -> Self {
        let data_width = x_max - x_min;
        let data_height = y_max - y_min;
        let safe_ratio = target_ratio.max(1.0e-6);
        let data_ratio = data_width / data_height.max(1e-12);

        if data_ratio > safe_ratio {
            let new_height = data_width / safe_ratio;
            let pad_y = (new_height - data_height) / 2.0;
            Self {
                x_min,
                x_max,
                y_min: y_min - pad_y,
                y_max: y_max + pad_y,
            }
        } else {
            let new_width = data_height * safe_ratio;
            let pad_x = (new_width - data_width) / 2.0;
            Self {
                x_min: x_min - pad_x,
                x_max: x_max + pad_x,
                y_min,
                y_max,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LatLonStats {
    min_lat: f64,
    max_lat: f64,
    center_lon: f64,
    lon_span: f64,
}

impl LatLonStats {
    fn from_grid(lat_deg: &[f32], lon_deg: &[f32]) -> Option<Self> {
        let mut min_lat = f64::INFINITY;
        let mut max_lat = f64::NEG_INFINITY;

        for &lat in lat_deg {
            let lat = lat as f64;
            if !lat.is_finite() {
                continue;
            }
            min_lat = min_lat.min(lat);
            max_lat = max_lat.max(lat);
        }

        if !min_lat.is_finite() || !max_lat.is_finite() {
            return None;
        }

        Some(Self {
            min_lat,
            max_lat,
            center_lon: circular_mean_longitude_deg(lon_deg)?,
            lon_span: minimal_longitude_span_deg(lon_deg)?,
        })
    }
}

fn minimal_longitude_span_deg(lon_deg: &[f32]) -> Option<f64> {
    let mut lon: Vec<f64> = lon_deg
        .iter()
        .filter_map(|&value| {
            let value = value as f64;
            value
                .is_finite()
                .then_some(normalize_longitude_positive_deg(value))
        })
        .collect();

    if lon.is_empty() {
        return None;
    }
    lon.sort_by(f64::total_cmp);
    lon.dedup_by(|a, b| (*a - *b).abs() < 1.0e-9);
    if lon.len() == 1 {
        return Some(0.0);
    }

    let mut max_gap: f64 = 0.0;
    for pair in lon.windows(2) {
        max_gap = max_gap.max(pair[1] - pair[0]);
    }
    max_gap = max_gap.max(lon[0] + 360.0 - lon[lon.len() - 1]);
    Some((360.0 - max_gap).clamp(0.0, 360.0))
}

fn inferred_standard_parallels(min_lat: f64, max_lat: f64, lat_span: f64) -> (f64, f64) {
    if min_lat < 0.0 && max_lat > 0.0 {
        let dominant = if max_lat.abs() >= min_lat.abs() {
            max_lat
        } else {
            min_lat
        };
        let tangent = stabilize_reference_latitude((dominant * 0.75).max(5.0 * dominant.signum()));
        return (tangent, tangent);
    }

    if lat_span < 6.0 {
        let tangent = stabilize_reference_latitude((min_lat + max_lat) / 2.0);
        return (tangent, tangent);
    }

    let inset = (lat_span / 6.0).clamp(2.0, 12.0);
    let sp1 = stabilize_reference_latitude(min_lat + inset);
    let sp2 = stabilize_reference_latitude(max_lat - inset);
    if (sp2 - sp1).abs() < 0.25 {
        (sp1, sp1)
    } else {
        (sp1, sp2)
    }
}

fn latitude_midpoint_deg(lat_deg: &[f32]) -> Option<f64> {
    let mut min_lat = f64::INFINITY;
    let mut max_lat = f64::NEG_INFINITY;
    for &lat in lat_deg {
        let lat = lat as f64;
        if !lat.is_finite() {
            continue;
        }
        min_lat = min_lat.min(lat);
        max_lat = max_lat.max(lat);
    }
    (min_lat.is_finite() && max_lat.is_finite()).then_some((min_lat + max_lat) / 2.0)
}

fn circular_mean_longitude_deg(lon_deg: &[f32]) -> Option<f64> {
    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;
    let mut count = 0usize;
    for &lon in lon_deg {
        let lon = lon as f64;
        if !lon.is_finite() {
            continue;
        }
        let lambda = lon * DEG2RAD;
        sin_sum += lambda.sin();
        cos_sum += lambda.cos();
        count += 1;
    }
    if count == 0 {
        return None;
    }
    if sin_sum.abs() < 1.0e-9 && cos_sum.abs() < 1.0e-9 {
        return Some(0.0);
    }
    Some(normalize_longitude_deg(sin_sum.atan2(cos_sum) * RAD2DEG))
}

fn stabilize_latitude(lat_deg: f64) -> f64 {
    lat_deg.clamp(-89.999, 89.999)
}

fn stabilize_reference_latitude(lat_deg: f64) -> f64 {
    let clamped = lat_deg.clamp(-85.0, 85.0);
    if clamped.abs() < 1.0 {
        10.0_f64.copysign(if clamped < 0.0 { -1.0 } else { 1.0 })
    } else {
        clamped
    }
}

fn normalize_longitude_deg(lon_deg: f64) -> f64 {
    let mut lon = lon_deg % 360.0;
    if lon > 180.0 {
        lon -= 360.0;
    } else if lon <= -180.0 {
        lon += 360.0;
    }
    lon
}

fn normalize_longitude_positive_deg(lon_deg: f64) -> f64 {
    let mut lon = lon_deg % 360.0;
    if lon < 0.0 {
        lon += 360.0;
    }
    lon
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_spec_round_trips_through_core_grid_projection() {
        let spec = ProjectionSpec::Mercator {
            latitude_of_true_scale_deg: 25.0,
            central_meridian_deg: -90.0,
        };
        let core: core::GridProjection = spec.clone().into();
        let round_trip = ProjectionSpec::from(core);
        assert_eq!(round_trip, spec);
    }

    #[test]
    fn inferred_projection_uses_region_centered_lambert() {
        let lat = vec![25.0, 25.0, 50.0, 50.0];
        let lon = vec![-124.0, -66.0, -124.0, -66.0];
        let inferred = ProjectionSpec::infer_from_latlon_grid(&lat, &lon)
            .expect("finite grid should infer a fallback projection");

        match inferred {
            ProjectionSpec::LambertConformal {
                standard_parallel_1_deg,
                standard_parallel_2_deg,
                central_meridian_deg,
            } => {
                assert!(standard_parallel_1_deg > 25.0);
                assert!(standard_parallel_2_deg < 50.0);
                assert!(central_meridian_deg < -90.0);
                assert!(central_meridian_deg > -100.0);
            }
            other => panic!("expected inferred Lambert conformal, got {other:?}"),
        }
    }

    #[test]
    fn inferred_projection_uses_geographic_for_global_latlon_grid() {
        let mut lat = Vec::new();
        let mut lon = Vec::new();
        for row_lat in [-90.0f32, 0.0, 90.0] {
            for col_lon in (0..360).step_by(30) {
                lat.push(row_lat);
                lon.push(col_lon as f32);
            }
        }

        let inferred = ProjectionSpec::infer_from_latlon_grid(&lat, &lon)
            .expect("finite global grid should infer a fallback projection");
        assert_eq!(inferred, ProjectionSpec::Geographic);
    }

    #[test]
    fn robinson_projection_softens_global_pole_width() {
        let spec = ProjectionSpec::Robinson {
            central_meridian_deg: 0.0,
        };
        let projector = spec
            .build_projector(None, None, &[0.0], &[0.0])
            .expect("robinson projector");
        let (equator_x, _) = projector.project(0.0, 90.0);
        let (high_lat_x, _) = projector.project(80.0, 90.0);
        assert!(high_lat_x.abs() < equator_x.abs());
    }

    #[test]
    fn minimal_longitude_span_handles_antimeridian_domains() {
        let lon = vec![176.0, 179.0, -179.0, -178.0];
        let span = minimal_longitude_span_deg(&lon).expect("finite span");
        assert!(span < 7.0, "expected a small antimeridian span, got {span}");
    }

    #[test]
    fn geographic_projector_centers_longitudes() {
        let spec = ProjectionSpec::Geographic;
        let projector = spec
            .build_projector(None, None, &[40.0, 41.0], &[-110.0, -90.0])
            .expect("geographic projector");
        let (x_west, y) = projector.project(40.0, -110.0);
        let (x_east, _) = projector.project(40.0, -90.0);
        assert!(x_west < 0.0);
        assert!(x_east > 0.0);
        assert_eq!(y, 40.0);
    }

    #[test]
    fn southern_hemisphere_lambert_round_trips() {
        let spec = ProjectionSpec::LambertConformal {
            standard_parallel_1_deg: -39.0,
            standard_parallel_2_deg: -15.0,
            central_meridian_deg: 133.0,
        };
        let projector = spec
            .build_projector(Some(-27.0), None, &[-45.0, -9.0], &[111.0, 156.0])
            .expect("southern Lambert projector");

        for (lat, lon) in [(-12.5, 130.0), (-27.0, 133.0), (-37.8, 145.0)] {
            let (x, y) = projector.project(lat, lon);
            let (actual_lat, actual_lon) = projector.unproject(x, y).expect("round trip");
            assert!((actual_lat - lat).abs() < 1.0e-8);
            assert!((normalize_longitude_deg(actual_lon - lon)).abs() < 1.0e-8);
        }
    }

    #[test]
    fn southern_hemisphere_albers_round_trips() {
        let spec = ProjectionSpec::AlbersEqualArea {
            standard_parallel_1_deg: -40.0,
            standard_parallel_2_deg: -18.0,
            central_meridian_deg: 133.0,
            latitude_of_origin_deg: -27.0,
        };
        let projector = spec
            .build_projector(None, None, &[-45.0, -9.0], &[111.0, 156.0])
            .expect("southern Albers projector");

        for (lat, lon) in [(-12.5, 130.0), (-27.0, 133.0), (-37.8, 145.0)] {
            let (x, y) = projector.project(lat, lon);
            let (actual_lat, actual_lon) = projector.unproject(x, y).expect("round trip");
            assert!((actual_lat - lat).abs() < 1.0e-8);
            assert!((normalize_longitude_deg(actual_lon - lon)).abs() < 1.0e-8);
        }
    }

    #[test]
    fn projected_families_return_finite_points() {
        let lat = [40.0f32, 42.0];
        let lon = [-110.0f32, -100.0];
        let specs = [
            ProjectionSpec::LambertConformal {
                standard_parallel_1_deg: 33.0,
                standard_parallel_2_deg: 45.0,
                central_meridian_deg: -97.0,
            },
            ProjectionSpec::AlbersEqualArea {
                standard_parallel_1_deg: 29.5,
                standard_parallel_2_deg: 45.5,
                central_meridian_deg: -96.0,
                latitude_of_origin_deg: 23.0,
            },
            ProjectionSpec::Mercator {
                latitude_of_true_scale_deg: 25.0,
                central_meridian_deg: -95.0,
            },
            ProjectionSpec::PolarStereographic {
                true_latitude_deg: 60.0,
                central_meridian_deg: -105.0,
                south_pole_on_projection_plane: false,
            },
        ];

        for spec in specs {
            let projector = spec
                .build_projector(None, None, &lat, &lon)
                .expect("supported projection");
            let (x, y) = projector.project(40.0, -105.0);
            assert!(x.is_finite());
            assert!(y.is_finite());
        }
    }

    #[test]
    fn unsupported_projection_template_is_rejected() {
        let spec = ProjectionSpec::Other { template: 1234 };
        let error = spec
            .build_projector(None, None, &[40.0], &[-100.0])
            .unwrap_err();
        assert!(error.contains("not supported"));
    }
}
