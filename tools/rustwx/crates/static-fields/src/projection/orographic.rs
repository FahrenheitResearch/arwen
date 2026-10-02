//! WPS default-REAL coordinates for sub-grid orographic statistics.
//!
//! The established terrain sampler retains its qualified arithmetic.
//! Orographic source stencils use WPS's scalar single-precision map state,
//! including its single-precision PI constants, without NumPy ULP nudges.
//!
//! Every transcendental here comes from rw-libm, never from the platform C
//! library: glibc 2.39, glibc 2.43 and the Windows UCRT return different
//! last bits for the same `f32::powf` or `f32::cos`, and the Lambert
//! inverse's cancellation turned one such bit into hundreds of ULP, so these
//! terrain-drag statics differed by platform (public CI run 37036982597).
//! rw-libm returns glibc 2.43's bits, the library the WPS oracle was built
//! against, on every platform.
use super::{ProjectedGrid, ProjectionKind};
use crate::error::{Result, StaticError};
use rw_libm::{acos, asin, atan2f, atanf, cosf, expf, log10f, logf, powf, sinf, tanf};

const PI: f32 = std::f32::consts::PI;
const RAD: f32 = PI / 180.0;
const DEG: f32 = 180.0 / PI;

pub(crate) struct OrographicProjection<'g> {
    grid: &'g ProjectedGrid,
    hemi: f32,
    cone: f32,
    rebydx: f32,
    polei: f32,
    polej: f32,
    rsw: f32,
    dlon: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::GridSpec;

    #[test]
    fn orographic_tangent_and_southern_coordinates_match_wps_bits() {
        let data = include_bytes!("../../golden/orographic/mesh-edge-real.bin");
        let inverse =
            include_bytes!("../../golden/orographic/inverse-real.bin");
        let mut inverse_word = 150usize;
        let mut word = 0usize;
        for (kind, lat, lon, stand, tl1, tl2) in [
            (ProjectionKind::Lambert, 35.0, -90.0, -90.0, 30.0, 30.0),
            (ProjectionKind::Lambert, -35.0, 18.0, 18.0, -30.0, -60.0),
            (ProjectionKind::Mercator, -15.0, 140.0, 140.0, -20.0, -20.0),
            (ProjectionKind::Polar, -70.0, 15.0, 0.0, -60.0, -60.0),
        ] {
            let grid = ProjectedGrid::new(GridSpec {
                kind,
                ref_lat: lat,
                ref_lon: lon,
                truelat1: tl1,
                truelat2: tl2,
                stand_lon: stand,
                dx: 3000.0,
                dy: 3000.0,
                e_we: 17,
                e_sn: 17,
                known_x: 8.5,
                known_y: 8.5,
                moad_cen_lat: lat,
                moad_cen_lon: lon,
                lat_deg: vec![],
                lon0_deg: 0.0,
                dlon_deg: 0.0,
            })
            .unwrap();
            let p = OrographicProjection::new(&grid).unwrap();
            for j in 1..=16 {
                for i in 1..=16 {
                    let (lat, lon) = p.ij_to_latlon(i as f32, j as f32);
                    if [1, 8, 16].contains(&i) && [1, 8, 16].contains(&j) {
                        let (x, y) = p.latlon_to_ij(&grid, lat, lon);
                        for got in [x, y] {
                            let expected = u32::from_le_bytes(
                                inverse
                                    [inverse_word * 4..(inverse_word + 1) * 4]
                                    .try_into()
                                    .unwrap(),
                            );
                            assert_eq!(
                                got.to_bits(),
                                expected,
                                "inverse {kind:?} {i},{j}"
                            );
                            inverse_word += 1;
                        }
                    }
                    for got in [lat, lon] {
                        let expected = u32::from_le_bytes(
                            data[word * 4..(word + 1) * 4].try_into().unwrap(),
                        );
                        assert_eq!(got.to_bits(), expected, "{kind:?} {i},{j}");
                        word += 1;
                    }
                }
            }
        }
    }

    #[test]
    fn orographic_coordinates_match_wps_real_oracle() {
        let data = include_bytes!("../../golden/orographic/mesh-real.bin");
        let inverse =
            include_bytes!("../../golden/orographic/inverse-real.bin");
        let mut inverse_word = 0usize;
        let mut word = 0usize;
        let mut different = 0usize;
        let mut maximum = 0u32;
        for (kind, lat, lon, kx, ky, stand, tl1, tl2) in [
            (
                ProjectionKind::Lambert,
                34.2,
                -118.2,
                75.5,
                75.5,
                -118.2,
                30.0,
                60.0,
            ),
            (
                ProjectionKind::Mercator,
                12.5,
                140.0,
                8.5,
                6.5,
                140.0,
                20.0,
                20.0,
            ),
            (ProjectionKind::Polar, 70.0, 15.0, 8.5, 6.5, 0.0, 60.0, 60.0),
        ] {
            let before = different;
            let grid = ProjectedGrid::new(GridSpec {
                kind,
                ref_lat: lat,
                ref_lon: lon,
                truelat1: tl1,
                truelat2: tl2,
                stand_lon: stand,
                dx: 3000.0,
                dy: 3000.0,
                e_we: 151,
                e_sn: 151,
                known_x: kx,
                known_y: ky,
                moad_cen_lat: lat,
                moad_cen_lon: lon,
                lat_deg: vec![],
                lon0_deg: 0.0,
                dlon_deg: 0.0,
            })
            .unwrap();
            let p = OrographicProjection::new(&grid).unwrap();
            println!(
                "{kind:?} state {:?}",
                [p.polei, p.polej, p.rebydx, p.rsw, p.cone, RAD, DEG]
            );
            for j in 1..=150 {
                for i in 1..=150 {
                    let (lat, lon) = p.ij_to_latlon(i as f32, j as f32);
                    if (i - 1) % 30 == 0 && (j - 1) % 30 == 0 {
                        let (x, y) = p.latlon_to_ij(&grid, lat, lon);
                        for got in [x, y] {
                            let expected = u32::from_le_bytes(
                                inverse
                                    [inverse_word * 4..(inverse_word + 1) * 4]
                                    .try_into()
                                    .unwrap(),
                            );
                            assert_eq!(
                                got.to_bits(),
                                expected,
                                "inverse {kind:?} {i},{j}"
                            );
                            inverse_word += 1;
                        }
                    }
                    for got in [lat, lon] {
                        let expected = u32::from_le_bytes(
                            data[word * 4..word * 4 + 4].try_into().unwrap(),
                        );
                        let d = got.to_bits().abs_diff(expected);
                        if d > 0 {
                            if different < 8 {
                                println!(
                                    "{kind:?} {i},{j}: {got:?} expected {:?} ulp {d}",
                                    f32::from_bits(expected)
                                );
                            }
                            different += 1;
                        }
                        maximum = maximum.max(d);
                        word += 1;
                    }
                }
            }
            if kind != ProjectionKind::Polar {
                assert_eq!(
                    different, before,
                    "{kind:?} WPS coordinate bits changed"
                );
            }
        }
        println!(
            "orographic WPS coordinates: {different} differing of {word}, maximum ULP {maximum}"
        );
        assert_eq!(different, 0, "WPS coordinate bits changed");
        assert_eq!(maximum, 0);
    }
}

impl<'g> OrographicProjection<'g> {
    pub fn new(grid: &'g ProjectedGrid) -> Result<Self> {
        if let Some((base, _)) = &grid.translation {
            return Self::new(base);
        }
        let s = &grid.sampling.spec;
        let h = if s.truelat1 < 0.0 { -1.0 } else { 1.0 };
        let tl1 = s.truelat1 as f32;
        let tl2 = s.truelat2 as f32;
        let re = 6_370_000.0f32 / s.dx as f32;
        let mut p = Self {
            grid,
            hemi: h,
            cone: 0.0,
            rebydx: re,
            polei: 0.0,
            polej: 0.0,
            rsw: 0.0,
            dlon: 0.0,
        };
        match s.kind {
            ProjectionKind::Lambert => {
                p.cone = if (tl1 - tl2).abs() > 0.1 {
                    (log10f(cosf(tl1 * RAD)) - log10f(cosf(tl2 * RAD)))
                        / (log10f(tanf((45.0 - tl1.abs() / 2.0) * RAD))
                            - log10f(tanf((45.0 - tl2.abs() / 2.0) * RAD)))
                } else {
                    sinf(tl1.abs() * RAD)
                };
                let mut dl = s.ref_lon as f32 - s.stand_lon as f32;
                if dl > 180.0 {
                    dl -= 360.0;
                }
                if dl < -180.0 {
                    dl += 360.0;
                }
                p.rsw = re * cosf(tl1 * RAD) / p.cone
                    * powf(
                        tanf((90.0 * h - s.ref_lat as f32) * RAD / 2.0)
                            / tanf((90.0 * h - tl1) * RAD / 2.0),
                        p.cone,
                    );
                let a = p.cone * (dl * RAD);
                p.polei = h * s.known_x as f32 - h * p.rsw * sinf(a);
                p.polej = h * s.known_y as f32 + p.rsw * cosf(a);
            }
            ProjectionKind::Mercator => {
                p.dlon = s.dx as f32 / (6_370_000.0f32 * cosf(RAD * tl1));
                if s.ref_lat != 0.0 {
                    p.rsw =
                        logf(tanf(0.5 * ((s.ref_lat as f32 + 90.0) * RAD)))
                            / p.dlon;
                }
            }
            ProjectionKind::Polar => {
                let top = 1.0 + h * sinf(tl1 * RAD);
                let a = s.ref_lat as f32 * RAD;
                p.rsw = re * cosf(a) * top / (1.0 + h * sinf(a));
                let a = (s.ref_lon as f32 - (s.stand_lon as f32 + 90.0)) * RAD;
                p.polei = s.known_x as f32 - p.rsw * cosf(a);
                p.polej = s.known_y as f32 - h * p.rsw * sinf(a);
            }
            ProjectionKind::Rows => {
                return Err(StaticError::Invalid(
                    "orographic WPS fields need a projected grid".into(),
                ));
            }
        }
        Ok(p)
    }

    pub fn ij_to_latlon(&self, x: f32, y: f32) -> (f32, f32) {
        let s = &self.grid.sampling.spec;
        let h = self.hemi;
        match s.kind {
            ProjectionKind::Lambert => {
                let xx = h * x - self.polei;
                let yy = self.polej - h * y;
                let r2 = xx * xx + yy * yy;
                if r2 == 0.0 {
                    return (h * 90.0, s.stand_lon as f32);
                }
                let r = r2.sqrt() / self.rebydx;
                let mut lon = (s.stand_lon as f32
                    + DEG * atan2f(h * xx, yy) / self.cone
                    + 360.0)
                    % 360.0;
                let c1 = (90.0 - h * s.truelat1 as f32) * RAD;
                let c2 = (90.0 - h * s.truelat2 as f32) * RAD;
                let chi = if c1 == c2 {
                    2.0 * atanf(
                        powf(r / tanf(c1), 1.0 / self.cone) * tanf(c1 * 0.5),
                    )
                } else {
                    2.0 * atanf(
                        powf(r * self.cone / sinf(c1), 1.0 / self.cone)
                            * tanf(c1 * 0.5),
                    )
                };
                if lon > 180.0 {
                    lon -= 360.0;
                }
                if lon < -180.0 {
                    lon += 360.0;
                }
                ((90.0 - chi * DEG) * h, lon)
            }
            ProjectionKind::Mercator => {
                let lat = 2.0
                    * atanf(expf(self.dlon * (self.rsw + y - s.known_y as f32)))
                    * DEG
                    - 90.0;
                let mut lon =
                    (x - s.known_x as f32) * self.dlon * DEG + s.ref_lon as f32;
                if lon > 180.0 {
                    lon -= 360.0;
                }
                if lon < -180.0 {
                    lon += 360.0;
                }
                (lat, lon)
            }
            ProjectionKind::Polar => {
                let xx = x - self.polei;
                let yy = (y - self.polej) * h;
                // WPS assigns f32 squares to REAL(HIGH) r2 and gi2, then
                // evaluates their ratio and inverse trigonometry in f64.
                let r2 = (xx * xx + yy * yy) as f64;
                let reflon = s.stand_lon as f32 + 90.0;
                if r2 == 0.0 {
                    return (h * 90.0, reflon);
                }
                let scale =
                    self.rebydx * (1.0 + h * sinf(s.truelat1 as f32 * RAD));
                let gi2 = powf(scale, 2.0) as f64;
                let lat = ((DEG * h) as f64 * asin((gi2 - r2) / (gi2 + r2)))
                    as f32;
                // WPS clamps with REAL(HIGH) literals, so ACOS is evaluated
                // in double precision and assigned back to default REAL.
                let a =
                    acos(((xx as f64) / r2.sqrt()).clamp(-1.0, 1.0)) as f32;
                let mut lon = if yy > 0.0 {
                    reflon + DEG * a
                } else {
                    reflon - DEG * a
                };
                if lon > 180.0 {
                    lon -= 360.0;
                }
                if lon < -180.0 {
                    lon += 360.0;
                }
                (lat, lon)
            }
            ProjectionKind::Rows => unreachable!(),
        }
    }

    pub fn for_grid(&self, grid: &ProjectedGrid, x: f32, y: f32) -> (f32, f32) {
        let (x, y) = if let Some((_, (di, dj))) = &grid.translation {
            (x + *di as f32, y + *dj as f32)
        } else {
            (x, y)
        };
        self.ij_to_latlon(x, y)
    }

    pub fn latlon_to_ij(
        &self,
        grid: &ProjectedGrid,
        lat: f32,
        lon: f32,
    ) -> (f32, f32) {
        let s = &self.grid.sampling.spec;
        let h = self.hemi;
        let (x, y) = match s.kind {
            ProjectionKind::Lambert => {
                let mut dl = lon - s.stand_lon as f32;
                if dl > 180.0 {
                    dl -= 360.0;
                }
                if dl < -180.0 {
                    dl += 360.0;
                }
                let rm = self.rebydx * cosf(s.truelat1 as f32 * RAD)
                    / self.cone
                    * powf(
                        tanf((90.0 * h - lat) * RAD / 2.0)
                            / tanf((90.0 * h - s.truelat1 as f32) * RAD / 2.0),
                        self.cone,
                    );
                let a = self.cone * (dl * RAD);
                (
                    h * (self.polei + h * rm * sinf(a)),
                    h * (self.polej - rm * cosf(a)),
                )
            }
            ProjectionKind::Mercator => {
                let mut dl = lon - s.ref_lon as f32;
                if dl > 180.0 {
                    dl -= 360.0;
                }
                if dl < -180.0 {
                    dl += 360.0;
                }
                (
                    s.known_x as f32 + dl / (self.dlon * DEG),
                    s.known_y as f32
                        + logf(tanf(0.5 * ((lat + 90.0) * RAD))) / self.dlon
                        - self.rsw,
                )
            }
            ProjectionKind::Polar => {
                let a = lat * RAD;
                let rm = self.rebydx
                    * cosf(a)
                    * (1.0 + h * sinf(s.truelat1 as f32 * RAD))
                    / (1.0 + h * sinf(a));
                let a = (lon - (s.stand_lon as f32 + 90.0)) * RAD;
                (self.polei + rm * cosf(a), self.polej + h * rm * sinf(a))
            }
            ProjectionKind::Rows => unreachable!(),
        };
        if let Some((_, (di, dj))) = &grid.translation {
            (x - *di as f32, y - *dj as f32)
        } else {
            (x, y)
        }
    }
}
