//! WPS default-REAL coordinates for sub-grid orographic statistics.
//!
//! The established terrain sampler retains its qualified arithmetic.
//! Orographic source stencils use WPS's scalar single-precision map state,
//! including its single-precision PI constants, without NumPy ULP nudges.
use super::{ProjectedGrid, ProjectionKind};
use crate::error::{Result, StaticError};

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
                    ((tl1 * RAD).cos().log10() - (tl2 * RAD).cos().log10())
                        / (((45.0 - tl1.abs() / 2.0) * RAD).tan().log10()
                            - ((45.0 - tl2.abs() / 2.0) * RAD).tan().log10())
                } else {
                    (tl1.abs() * RAD).sin()
                };
                let mut dl = s.ref_lon as f32 - s.stand_lon as f32;
                if dl > 180.0 {
                    dl -= 360.0;
                }
                if dl < -180.0 {
                    dl += 360.0;
                }
                p.rsw = re * (tl1 * RAD).cos() / p.cone
                    * (((90.0 * h - s.ref_lat as f32) * RAD / 2.0).tan()
                        / ((90.0 * h - tl1) * RAD / 2.0).tan())
                    .powf(p.cone);
                let a = p.cone * (dl * RAD);
                p.polei = h * s.known_x as f32 - h * p.rsw * a.sin();
                p.polej = h * s.known_y as f32 + p.rsw * a.cos();
            }
            ProjectionKind::Mercator => {
                p.dlon = s.dx as f32 / (6_370_000.0f32 * (RAD * tl1).cos());
                if s.ref_lat != 0.0 {
                    p.rsw =
                        (0.5 * ((s.ref_lat as f32 + 90.0) * RAD)).tan().ln()
                            / p.dlon;
                }
            }
            ProjectionKind::Polar => {
                let top = 1.0 + h * (tl1 * RAD).sin();
                let a = s.ref_lat as f32 * RAD;
                p.rsw = re * a.cos() * top / (1.0 + h * a.sin());
                let a = (s.ref_lon as f32 - (s.stand_lon as f32 + 90.0)) * RAD;
                p.polei = s.known_x as f32 - p.rsw * a.cos();
                p.polej = s.known_y as f32 - h * p.rsw * a.sin();
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
                    + DEG * (h * xx).atan2(yy) / self.cone
                    + 360.0)
                    % 360.0;
                let c1 = (90.0 - h * s.truelat1 as f32) * RAD;
                let c2 = (90.0 - h * s.truelat2 as f32) * RAD;
                let chi = if c1 == c2 {
                    2.0 * ((r / c1.tan()).powf(1.0 / self.cone)
                        * (c1 * 0.5).tan())
                    .atan()
                } else {
                    2.0 * ((r * self.cone / c1.sin()).powf(1.0 / self.cone)
                        * (c1 * 0.5).tan())
                    .atan()
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
                    * (self.dlon * (self.rsw + y - s.known_y as f32))
                        .exp()
                        .atan()
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
                    self.rebydx * (1.0 + h * (s.truelat1 as f32 * RAD).sin());
                let gi2 = scale.powf(2.0) as f64;
                let lat = ((DEG * h) as f64 * ((gi2 - r2) / (gi2 + r2)).asin())
                    as f32;
                // WPS clamps with REAL(HIGH) literals, so ACOS is evaluated
                // in double precision and assigned back to default REAL.
                let a =
                    ((xx as f64) / r2.sqrt()).clamp(-1.0, 1.0).acos() as f32;
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
                let rm = self.rebydx * (s.truelat1 as f32 * RAD).cos()
                    / self.cone
                    * (((90.0 * h - lat) * RAD / 2.0).tan()
                        / ((90.0 * h - s.truelat1 as f32) * RAD / 2.0).tan())
                    .powf(self.cone);
                let a = self.cone * (dl * RAD);
                (
                    h * (self.polei + h * rm * a.sin()),
                    h * (self.polej - rm * a.cos()),
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
                        + (0.5 * ((lat + 90.0) * RAD)).tan().ln() / self.dlon
                        - self.rsw,
                )
            }
            ProjectionKind::Polar => {
                let a = lat * RAD;
                let rm = self.rebydx
                    * a.cos()
                    * (1.0 + h * (s.truelat1 as f32 * RAD).sin())
                    / (1.0 + h * a.sin());
                let a = (lon - (s.stand_lon as f32 + 90.0)) * RAD;
                (self.polei + rm * a.cos(), self.polej + h * rm * a.sin())
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
