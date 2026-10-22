//! Native-column heat/fire diagnostics. See DERIVED-DIAGNOSTICS-NOTICES.txt.
//! SI inputs; Haines dimensionless, instantaneous HDW hPa m/s, WBGT degC.
use rayon::prelude::*;
/// Heat quantities reconstructed from an ASOS screen temperature/dewpoint.
/// Matches the model surface products' NWS heat index and NWS apparent
/// temperature (heat index / wind chill / air temperature) convention.
/// ASOS alone cannot validate solar WBGT: radiation is not observed by METAR.
pub fn screen_heat(temperature_k: f64, dewpoint_k: f64, wind_ms: Option<f64>) -> (f64, f64) {
    if !temperature_k.is_finite() || !dewpoint_k.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let tc = temperature_k - 273.15;
    let rh = metrust::calc::thermo::relative_humidity_from_dewpoint(tc, dewpoint_k - 273.15);
    let hi = metrust::calc::atmo::heat_index(tc, rh);
    let apparent = wind_ms
        .filter(|v| v.is_finite() && *v >= 0.0)
        .map(|v| metrust::calc::atmo::apparent_temperature(tc, rh, v))
        .unwrap_or(f64::NAN);
    (hi, apparent)
}

#[derive(Clone, Copy)]
pub struct ThermoColumns<'a> {
    pub cells: usize,
    pub nz: usize,
    pub pressure_pa: &'a [f64],
    pub temperature_k: &'a [f64],
    pub vapor_kgkg: &'a [f64],
    pub height_agl_m: &'a [f64],
    pub interface_agl_m: &'a [f64],
}
impl ThermoColumns<'_> {
    pub fn validate(&self) -> Result<(), String> {
        let n = self
            .cells
            .checked_mul(self.nz)
            .ok_or("column shape overflow")?;
        if self.cells == 0
            || self.nz < 2
            || [
                self.pressure_pa,
                self.temperature_k,
                self.vapor_kgkg,
                self.height_agl_m,
            ]
            .iter()
            .any(|a| a.len() != n)
            || self
                .nz
                .checked_add(1)
                .and_then(|z| z.checked_mul(self.cells))
                != Some(self.interface_agl_m.len())
        {
            return Err("columns require [nz,cells] fields and [nz+1,cells] interfaces".into());
        }
        Ok(())
    }
    /// No extrapolation under terrain or above the model top.
    pub fn at_pressure(&self, i: usize, target: f64, values: &[f64]) -> f64 {
        if i >= self.cells
            || values.len() != self.cells * self.nz
            || !target.is_finite()
            || target <= 0.0
        {
            return f64::NAN;
        }
        for k in 0..self.nz - 1 {
            let a = k * self.cells + i;
            let b = a + self.cells;
            let (p0, p1) = (self.pressure_pa[a], self.pressure_pa[b]);
            if !p0.is_finite() || !p1.is_finite() || p1 <= 0.0 || p0 <= p1 {
                return f64::NAN;
            }
            if p0 >= target && target >= p1 {
                let f = (p0 / target).ln() / (p0 / p1).ln();
                return values[a] * (1.0 - f) + values[b] * f;
            }
        }
        f64::NAN
    }
}
pub fn saturation_hpa(t: f64) -> f64 {
    6.1121 * (17.502 * (t - 273.15) / (t - 32.18)).exp()
}
pub fn vapor_hpa(p: f64, r: f64) -> f64 {
    p * 0.01 * r / (0.622 + r)
}
fn dewpoint_k(p: f64, r: f64) -> f64 {
    let e = vapor_hpa(p, r);
    if !e.is_finite() || e <= 0.0 {
        return f64::NAN;
    }
    let l = (e / 6.1121).ln();
    273.15 + 240.97 * l / (17.502 - l)
}
fn component(v: f64, a: f64, b: f64) -> f64 {
    if !v.is_finite() {
        f64::NAN
    } else if v < a {
        1.0
    } else if v < b {
        2.0
    } else {
        3.0
    }
}
/// Low moisture uses 850 hPa, although low stability starts at 950 hPa.
pub fn haines(c: ThermoColumns<'_>, variant: usize) -> Result<Vec<f64>, String> {
    c.validate()?;
    let (p0, p1, pm, a, b, d, e) = match variant {
        0 => (95000.0, 85000.0, 85000.0, 4.0, 8.0, 6.0, 10.0),
        1 => (85000.0, 70000.0, 85000.0, 6.0, 11.0, 6.0, 13.0),
        2 => (70000.0, 50000.0, 70000.0, 18.0, 22.0, 15.0, 21.0),
        _ => return Err("Haines variant must be 0, 1 or 2".into()),
    };
    Ok((0..c.cells)
        .into_par_iter()
        .map(|i| {
            let t0 = c.at_pressure(i, p0, c.temperature_k);
            let t1 = c.at_pressure(i, p1, c.temperature_k);
            let tm = c.at_pressure(i, pm, c.temperature_k);
            let r = c.at_pressure(i, pm, c.vapor_kgkg);
            if r < 0.0 {
                return f64::NAN;
            }
            component(t0 - t1, a, b) + component(tm - dewpoint_k(pm, r), d, e)
        })
        .collect())
}
/// Independent wind/VPD maxima below 500 m, including surface and 500 m
/// endpoint. Adjust each parcel dry adiabatically to surface pressure.
/// A daily HDW is the daily maximum of these instantaneous planes.
#[allow(clippy::too_many_arguments)]
pub fn hot_dry_windy(
    c: ThermoColumns<'_>,
    ps: &[f64],
    ts: &[f64],
    rs: &[f64],
    us: &[f64],
    vs: &[f64],
    u: &[f64],
    v: &[f64],
) -> Result<Vec<f64>, String> {
    c.validate()?;
    if [ps, ts, rs, us, vs].iter().any(|a| a.len() != c.cells)
        || [u, v].iter().any(|a| a.len() != c.cells * c.nz)
    {
        return Err("HDW shape mismatch".into());
    }
    Ok((0..c.cells)
        .into_par_iter()
        .map(|i| {
            if [ps[i], ts[i], rs[i], us[i], vs[i]]
                .iter()
                .any(|x| !x.is_finite())
                || ps[i] <= 0.0
                || ts[i] <= 0.0
                || rs[i] < 0.0
            {
                return f64::NAN;
            }
            let mut wind = us[i].hypot(vs[i]);
            let mut vpd = (saturation_hpa(ts[i]) - vapor_hpa(ps[i], rs[i])).max(0.0);
            let mut prev = (0.0, ps[i], ts[i], rs[i], us[i], vs[i]);
            for k in 0..c.nz {
                let j = k * c.cells + i;
                let (z, p, t, r, uk, vk) = (
                    c.height_agl_m[j],
                    c.pressure_pa[j],
                    c.temperature_k[j],
                    c.vapor_kgkg[j],
                    u[j],
                    v[j],
                );
                if [z, p, t, r, uk, vk].iter().any(|x| !x.is_finite())
                    || z < prev.0
                    || p <= 0.0
                    || t <= 0.0
                    || r < 0.0
                {
                    return f64::NAN;
                }
                let f = if z > 500.0 {
                    (500.0 - prev.0) / (z - prev.0)
                } else {
                    1.0
                };
                let lerp = |a: f64, b: f64| a + (b - a) * f;
                let pp = lerp(prev.1, p);
                let tt = lerp(prev.2, t);
                let rr = lerp(prev.3, r);
                wind = wind.max(lerp(prev.4, uk).hypot(lerp(prev.5, vk)));
                vpd = vpd.max(
                    (saturation_hpa(tt * (ps[i] / pp).powf(287.04 / 1004.0))
                        - vapor_hpa(ps[i], rr))
                    .max(0.0),
                );
                if z >= 500.0 {
                    return wind * vpd;
                }
                prev = (z, p, t, r, uk, vk);
            }
            f64::NAN
        })
        .collect())
}
const SB: f64 = 5.6696e-8;
const CP: f64 = 1003.5;
const RA: f64 = 8314.34 / 28.97;
const PR: f64 = CP / (CP + 1.25 * RA);
fn viscosity(t: f64) -> f64 {
    2.6693e-6 * (28.97 * t).sqrt() / (3.617 * 3.617 * (((t / 97.0 - 2.9) / 0.4) * (-0.034) + 1.048))
}
fn diffusivity(t: f64, p: f64) -> f64 {
    3.640e-4
        * (t / (132.0_f64 * 647.3).sqrt()).powf(2.334)
        * (36.4_f64 * 218.0).powf(1.0 / 3.0)
        * (132.0_f64 * 647.3).powf(5.0 / 12.0)
        * (1.0_f64 / 28.97 + 1.0 / 18.015).sqrt()
        / (p / 1013.25)
        * 1e-4
}
fn es(t: f64) -> f64 {
    1.004 * saturation_hpa(t)
}
fn convection(t: f64, p: f64, speed: f64, sphere: bool) -> f64 {
    let d = if sphere { 0.0508 } else { 0.007 };
    let re = speed.max(0.13) * (p * 100.0 / (RA * t)) * d / viscosity(t);
    let nu = if sphere {
        2.0 + 0.6 * re.sqrt() * PR.powf(0.3333)
    } else {
        0.281 * re.powf(0.6) * PR.powf(0.44)
    };
    nu * (CP + 1.25 * RA) * viscosity(t) / d
}
/// Rural SRDT power law from the reference; dt is T10m minus T2m.
pub fn sensor_wind_2m(w: f64, sw: f64, cza: f64, dt: f64) -> f64 {
    if [w, sw, cza, dt].iter().any(|v| !v.is_finite()) || w < 0.0 || sw < 0.0 {
        return f64::NAN;
    }
    let st = if cza > 0.0 {
        let j = if sw >= 925.0 {
            0
        } else if sw >= 675.0 {
            1
        } else if sw >= 175.0 {
            2
        } else {
            3
        };
        let i = if w >= 6.0 {
            4
        } else if w >= 5.0 {
            3
        } else if w >= 3.0 {
            2
        } else if w >= 2.0 {
            1
        } else {
            0
        };
        [
            [1, 1, 2, 4],
            [1, 2, 3, 4],
            [2, 2, 3, 4],
            [3, 3, 4, 4],
            [3, 4, 4, 4],
        ][i][j]
    } else if w >= 2.5 {
        4
    } else if dt >= 0.0 {
        6
    } else {
        5
    };
    (w * 0.2_f64.powf([0.07, 0.07, 0.10, 0.15, 0.35, 0.55][st - 1])).max(0.13)
}
/// Almanac solar position, UTC Unix seconds. Cosine zenith and distance AU.
pub fn solar_geometry(epoch: f64, lat: f64, lon: f64) -> (f64, f64) {
    if !epoch.is_finite() || !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
        return (f64::NAN, f64::NAN);
    }
    let days = epoch / 86400.0 + 2440587.5 - 2451545.0;
    let cent = ((epoch / 86400.0).floor() + 2440587.5 - 2451545.0) / 36525.0;
    let ut = epoch.rem_euclid(86400.0) / 3600.0;
    let g = (357.528 + 0.9856003 * days).to_radians();
    let l = (280.460 + 0.9856474 * days).to_radians();
    let obl = (23.439 - 4e-7 * days).to_radians();
    let e = l + (1.915 * g.sin() + 0.020 * (2.0 * g).sin()).to_radians();
    let ra = (obl.cos() * e.sin()).atan2(e.cos());
    let dec = (obl.sin() * e.sin()).asin();
    let gmst = (24110.54841 + cent * (8640184.812866 + cent * (0.093104 - cent * 6.2e-6))) / 3600.0;
    let ha = ((gmst + ut * 1.00273790934 + lon / 15.0) * 15.0).to_radians() - ra;
    let raw = (dec.sin() * lat.to_radians().sin() + dec.cos() * ha.cos() * lat.to_radians().cos())
        .clamp(-1.0, 1.0);
    let altitude = raw.asin().to_degrees();
    // Reference solarposition uses standard 1013.25 hPa / 15 C refraction.
    let correction = if altitude < -1.0 || altitude.abs() >= 89.99999 {
        0.0
    } else if altitude < 19.225 {
        (0.1594 + altitude * (0.0196 + 0.00002 * altitude)) * 1013.25
            / ((1.0 + altitude * (0.505 + 0.0845 * altitude)) * 288.0)
    } else {
        0.00452 * (1013.25 / 288.0) / altitude.to_radians().tan()
    };
    (
        (altitude + correction).to_radians().sin(),
        1.00014 - 0.01671 * g.cos() - 0.00014 * (2.0 * g).cos(),
    )
}
/// Liljegren 2008 sensor balances. Returns NaN on invalid input or
/// failure of either sensor to converge to 0.02 K within 50 iterations.
#[allow(clippy::too_many_arguments)]
pub fn wbgt_liljegren(t: f64, rh: f64, ppa: f64, w: f64, sw: f64, cza: f64, dist: f64) -> f64 {
    if [t, rh, ppa, w, sw, cza, dist]
        .iter()
        .any(|x| !x.is_finite())
        || !(180.0..=350.0).contains(&t)
        || !(0.0..=1.0).contains(&rh)
        || rh == 0.0
        || ppa <= 0.0
        || w < 0.0
        || sw < 0.0
        || dist <= 0.0
        || !(-1.0..=1.0).contains(&cza)
    {
        return f64::NAN;
    }
    let p = ppa * 0.01;
    let toa = if cza >= 0.00873 {
        1367.0 * cza / dist.powi(2)
    } else {
        0.0
    };
    let norm = if toa > 0.0 { (sw / toa).min(0.85) } else { 0.0 };
    let solar = if toa > 0.0 { norm * toa } else { sw };
    let fd = if norm > 0.0 {
        (3.0 - 1.34 * norm - 1.65 / norm).exp().clamp(0.0, 0.9)
    } else {
        0.0
    };
    let ea = rh * es(t);
    let l = (ea / (6.1121 * 1.004)).ln();
    let mut tw = 273.15 + 240.97 * l / (17.502 - l);
    let mut tg = t;
    let sky = 0.575 * ea.powf(0.143);
    let mut wet = None;
    let mut globe = None;
    for _ in 0..50 {
        if wet.is_none() {
            let tr = 0.5 * (tw + t);
            let h = convection(tr, p, w, false);
            let tan = if fd > 0.0 {
                (1.0 - cza * cza).max(0.0).sqrt() / cza
            } else {
                0.0
            };
            let fatm = SB * 0.95 * (0.5 * (sky * t.powi(4) + 0.999 * t.powi(4)) - tw.powi(4))
                + 0.6
                    * solar
                    * ((1.0 - fd) * (1.0 + 0.25 * 0.007 / 0.0254)
                        + fd * (tan / std::f64::consts::PI + 0.25 * 0.007 / 0.0254)
                        + 0.45);
            let sc = viscosity(tr) / (p * 100.0 / (RA * tr) * diffusivity(tr, p));
            let latent = (313.15 - tr) / 30.0 * (-71100.0) + 2.4073e6;
            let next = t - latent / (CP * 28.97 / 18.015) * (es(tw) - ea) / (p - es(tw))
                * (PR / sc).powf(0.56)
                + fatm / h;
            if (next - tw).abs() < 0.02 {
                wet = Some(next);
            }
            tw = 0.9 * tw + 0.1 * next;
        }
        if globe.is_none() {
            let h = convection(0.5 * (tg + t), p, w, true);
            let direct = if fd > 0.0 {
                fd * (1.0 / (2.0 * cza) - 1.0)
            } else {
                0.0
            };
            let next = (0.5 * (sky * t.powi(4) + 0.999 * t.powi(4)) - h / (SB * 0.95) * (tg - t)
                + solar / (2.0 * SB * 0.95) * 0.95 * (direct + 1.0 + 0.45))
                .powf(0.25);
            if (next - tg).abs() < 0.02 {
                globe = Some(next);
            }
            tg = 0.9 * tg + 0.1 * next;
        }
        if let (Some(wet), Some(globe)) = (wet, globe) {
            return 0.7 * (wet - 273.15) + 0.2 * (globe - 273.15) + 0.1 * (t - 273.15);
        }
    }
    f64::NAN
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn haines_uses_the_variant_pressure_levels_and_masks_terrain() {
        let p = [95000., 85000., 70000., 50000.];
        let t = [305., 300., 288., 270.];
        let r: Vec<_> = [280., 280., 283., 260.]
            .iter()
            .zip(p)
            .map(|(&td, p)| {
                let e = saturation_hpa(td) * 100.;
                0.622 * e / (p - e)
            })
            .collect();
        let z = [500., 1500., 3000., 5500.];
        let zi = [0., 1000., 2000., 4000., 7000.];
        let c = ThermoColumns {
            cells: 1,
            nz: 4,
            pressure_pa: &p,
            temperature_k: &t,
            vapor_kgkg: &r,
            height_agl_m: &z,
            interface_agl_m: &zi,
        };
        assert_eq!(haines(c, 0).unwrap(), vec![5.]);
        assert_eq!(haines(c, 1).unwrap(), vec![6.]);
        assert_eq!(haines(c, 2).unwrap(), vec![3.]);
        let elevated = [90000., 85000., 70000., 50000.];
        assert!(
            haines(
                ThermoColumns {
                    pressure_pa: &elevated,
                    ..c
                },
                0
            )
            .unwrap()[0]
                .is_nan()
        );
    }
    #[test]
    fn hdw_uses_independent_wind_and_vpd_maxima_and_the_500m_endpoint() {
        let p = [100000., 100000.];
        let t = [300., 306.];
        let q = [0., 0.];
        let z = [100., 600.];
        let zi = [0., 200., 1000.];
        let c = ThermoColumns {
            cells: 1,
            nz: 2,
            pressure_pa: &p,
            temperature_k: &t,
            vapor_kgkg: &q,
            height_agl_m: &z,
            interface_agl_m: &zi,
        };
        let v = hot_dry_windy(
            c,
            &[100000.],
            &[300.],
            &[0.],
            &[20.],
            &[0.],
            &[5., 1.],
            &[0., 0.],
        )
        .unwrap();
        // Wind maximum is at the surface; temperature at 500 m is 304.8 K.
        assert!((v[0] - 20. * saturation_hpa(304.8)).abs() < 1e-10);
        let short = [100., 400.];
        assert!(
            hot_dry_windy(
                ThermoColumns {
                    height_agl_m: &short,
                    ..c
                },
                &[100000.],
                &[300.],
                &[0.],
                &[20.],
                &[0.],
                &[5., 1.],
                &[0., 0.]
            )
            .unwrap()[0]
                .is_nan()
        );
    }
    /// Expected WBGT from unmodified public reference C sensor routines.
    /// The C float implementation and this f64 implementation agree within
    /// 0.002 C; the documented iteration tolerance is 0.02 K per sensor.
    #[test]
    fn independent_c_reference_balances() {
        let cases = [
            (303.15, 0.5, 101325.0, 1.0, 0.0, -0.5, 1.0, 24.15069370),
            (303.15, 0.5, 101325.0, 1.0, 800.0, 0.8, 1.0, 30.17953243),
            (313.15, 0.2, 99000.0, 0.13, 1000.0, 0.9, 1.02, 40.48149166),
            (308.15, 0.8, 101000.0, 3.0, 700.0, 0.5, 1.0, 35.81003342),
            (273.15, 0.8, 85000.0, 2.0, 0.0, -0.3, 1.0, -1.50114988),
            (303.15, 0.5, 101325.0, 10.0, 800.0, 0.8, 1.0, 26.38146725),
        ];
        for (t, rh, p, w, sw, cza, d, expected) in cases {
            let actual = wbgt_liljegren(t, rh, p, w, sw, cza, d);
            assert!(
                (actual - expected).abs() < 0.002,
                "WBGT {actual} != C {expected}"
            );
        }
    }
    #[test]
    fn radiation_and_missing_inputs() {
        let dark = wbgt_liljegren(303.15, 0.5, 101325.0, 1.0, 0.0, -0.5, 1.0);
        let sun = wbgt_liljegren(303.15, 0.5, 101325.0, 1.0, 800.0, 0.8, 1.0);
        assert!(sun > dark + 2.0);
        assert!(wbgt_liljegren(303.15, f64::NAN, 101325.0, 1.0, 800.0, 0.8, 1.0).is_nan());
    }
}

#[cfg(test)]
mod observation_heat_tests {
    use super::screen_heat;
    #[test]
    fn asos_missing_wind_does_not_become_calm_or_erase_heat_index() {
        let (hi, apparent) = screen_heat(308.15, 303.15, None);
        assert!(hi > 45.0 && hi < 60.0);
        assert!(apparent.is_nan());
        let (with_wind, apparent) = screen_heat(308.15, 303.15, Some(4.0));
        assert_eq!(hi, with_wind);
        assert_eq!(hi, apparent);
        assert!(screen_heat(308.15, f64::NAN, Some(4.0)).0.is_nan());
    }
}
