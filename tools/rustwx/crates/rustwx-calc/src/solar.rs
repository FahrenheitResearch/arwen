//! Solar products. Grid operations use Rayon; references: licenses/solar-sources.md.
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PvSystem {
    pub id: String,
    pub tracking: bool,
    pub tilt_deg: f64,
    pub azimuth_deg: f64,
    pub max_rotation_deg: f64,
    pub backtrack: bool,
    pub ground_coverage_ratio: f64,
    pub albedo: f64,
    pub dc_mw: f64,
    pub dc_ac_ratio: f64,
    pub inverter_efficiency: f64,
    pub gamma_per_c: f64,
    pub losses_fraction: f64,
    pub thermal_u0: f64,
    pub thermal_u1: f64,
    pub module_height_m: f64,
    pub ramp_fraction_per_minute: f64,
}
impl PvSystem {
    pub fn validate(&self) -> Result<(), String> {
        let finite = [
            self.tilt_deg,
            self.azimuth_deg,
            self.max_rotation_deg,
            self.ground_coverage_ratio,
            self.albedo,
            self.dc_mw,
            self.dc_ac_ratio,
            self.inverter_efficiency,
            self.gamma_per_c,
            self.losses_fraction,
            self.thermal_u0,
            self.thermal_u1,
            self.module_height_m,
            self.ramp_fraction_per_minute,
        ]
        .iter()
        .all(|v| v.is_finite());
        if !finite
            || self.id.is_empty()
            || !self
                .id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_')
            || !(0.0..=90.0).contains(&self.tilt_deg)
            || !(0.0..360.0).contains(&self.azimuth_deg)
            || !(0.0..=90.0).contains(&self.max_rotation_deg)
            || !(0.0..=1.0).contains(&self.albedo)
            || self.ground_coverage_ratio <= 0.0
            || self.ground_coverage_ratio > 1.0
            || self.dc_mw <= 0.0
            || self.dc_ac_ratio <= 0.0
            || self.inverter_efficiency <= 0.0
            || self.inverter_efficiency > 1.0
            || !(0.0..1.0).contains(&self.losses_fraction)
            || self.thermal_u0 <= 0.0
            || self.thermal_u1 < 0.0
            || self.module_height_m <= 0.0
            || self.gamma_per_c > 0.0
            || self.ramp_fraction_per_minute <= 0.0
        {
            return Err(format!("invalid PV system row {}", self.id));
        }
        Ok(())
    }
    pub fn ac_rating_mw(&self) -> f64 {
        self.dc_mw / self.dc_ac_ratio
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Sun {
    pub east: f64,
    pub north: f64,
    pub up: f64,
    pub extraterrestrial: f64,
}
/// NOAA fractional-year geometry, UTC, east-positive longitude, no refraction.
pub fn sun(latitude: f64, longitude: f64, day: u32, days_in_year: u32, utc_seconds: f64) -> Sun {
    let g = 2.0 * PI / f64::from(days_in_year)
        * (f64::from(day) - 1.0 + (utc_seconds / 3600.0 - 12.0) / 24.0);
    let eq = 229.18
        * (0.000075 + 0.001868 * g.cos()
            - 0.032077 * g.sin()
            - 0.014615 * (2.0 * g).cos()
            - 0.040849 * (2.0 * g).sin());
    let dec = 0.006918 - 0.399912 * g.cos() + 0.070257 * g.sin() - 0.006758 * (2.0 * g).cos()
        + 0.000907 * (2.0 * g).sin()
        - 0.002697 * (3.0 * g).cos()
        + 0.00148 * (3.0 * g).sin();
    let h = ((utc_seconds / 60.0 + eq + 4.0 * longitude) / 4.0 - 180.0).to_radians();
    let lat = latitude.to_radians();
    Sun {
        east: -dec.cos() * h.sin(),
        north: lat.cos() * dec.sin() - lat.sin() * dec.cos() * h.cos(),
        up: lat.sin() * dec.sin() + lat.cos() * dec.cos() * h.cos(),
        extraterrestrial: 1366.1
            * (1.00011
                + 0.034221 * g.cos()
                + 0.00128 * g.sin()
                + 0.000719 * (2.0 * g).cos()
                + 0.000077 * (2.0 * g).sin()),
    }
}
fn flux(v: f64) -> f64 {
    if v.is_finite() && (-5.0..=2500.0).contains(&v) {
        v.max(0.0)
    } else {
        f64::NAN
    }
}
/// Erbs 1982 hourly polynomial. Solar zenith >87 degrees is diffuse.
pub fn erbs(ghi: f64, sun: Sun) -> (f64, f64) {
    let ghi = flux(ghi);
    if !ghi.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    if sun.up <= 0.0 {
        return (0.0, 0.0);
    }
    if sun.up < 87.0_f64.to_radians().cos() {
        return (0.0, ghi);
    }
    let kt = (ghi / (sun.extraterrestrial * sun.up)).clamp(0.0, 1.0);
    let kd = if kt <= 0.22 {
        1.0 - 0.09 * kt
    } else if kt <= 0.8 {
        0.9511 - 0.1604 * kt + 4.388 * kt.powi(2) - 16.638 * kt.powi(3) + 12.336 * kt.powi(4)
    } else {
        0.165
    };
    let dhi = (ghi * kd).clamp(0.0, ghi);
    ((ghi - dhi) / sun.up, dhi)
}
pub fn clear_sky_ghi(sun: Sun) -> f64 {
    if sun.up > 0.0 {
        1098.0 * sun.up * (-0.059 / sun.up).exp()
    } else {
        0.0
    }
}
/// East/north/up normal. Tracker axis tilt is positive downward along azimuth.
pub fn normal(p: &PvSystem, sun: Sun) -> [f64; 3] {
    let (st, ct) = p.tilt_deg.to_radians().sin_cos();
    let (sa, ca) = p.azimuth_deg.to_radians().sin_cos();
    let n0 = [st * sa, st * ca, ct];
    if !p.tracking {
        return n0;
    }
    let b = [ca, -sa, 0.0];
    let mut t = (sun.east * b[0] + sun.north * b[1])
        .atan2(sun.east * n0[0] + sun.north * n0[1] + sun.up * n0[2]);
    if p.backtrack {
        let d = t.cos().abs() / p.ground_coverage_ratio;
        if d < 1.0 {
            t -= t.signum() * d.acos();
        }
    }
    t = t.clamp(
        -p.max_rotation_deg.to_radians(),
        p.max_rotation_deg.to_radians(),
    );
    let (s, c) = t.sin_cos();
    [n0[0] * c + b[0] * s, n0[1] * c + b[1] * s, n0[2] * c]
}
/// Isotropic sky and ground diffuse transposition; unshaded monofacial array.
pub fn poa(p: &PvSystem, s: Sun, ghi: f64, dni: f64, dhi: f64) -> f64 {
    if ![ghi, dni, dhi, s.up].iter().all(|v| v.is_finite()) {
        return f64::NAN;
    }
    if s.up <= 0.0 {
        return 0.0;
    }
    let n = normal(p, s);
    let inc = (n[0] * s.east + n[1] * s.north + n[2] * s.up).max(0.0);
    dni * inc + dhi * (1.0 + n[2]) * 0.5 + ghi * p.albedo * (1.0 - n[2]) * 0.5
}
/// PVWatts v5 DC eq 8 and inverter eqs 9-11, with explicit Faiman temperature.
/// Incident POA is effective irradiance; optical/soiling/wiring losses are metadata.
pub fn pvwatts(p: &PvSystem, poa: f64, air_c: f64, wind10: f64) -> (f64, f64) {
    if ![poa, air_c, wind10].iter().all(|v| v.is_finite())
        || poa < 0.0
        || wind10 < 0.0
        || !(-100.0..80.0).contains(&air_c)
    {
        return (f64::NAN, f64::NAN);
    }
    let wind = wind10 * (p.module_height_m / 10.0).powf(0.2);
    let cell = air_c + poa / (p.thermal_u0 + p.thermal_u1 * wind);
    let dc = (p.dc_mw * poa / 1000.0
        * (1.0 + p.gamma_per_c * (cell - 25.0))
        * (1.0 - p.losses_fraction))
        .max(0.0);
    let z = dc / (p.ac_rating_mw() / p.inverter_efficiency);
    let ac = if z > 0.0 {
        (p.inverter_efficiency / 0.9637 * (-0.0162 * z - 0.0059 / z + 0.9858) * dc)
            .clamp(0.0, p.ac_rating_mw())
    } else {
        0.0
    };
    (dc, ac)
}
pub struct Inputs<'a> {
    pub lat: &'a [f32],
    pub lon: &'a [f32],
    pub ghi: &'a [f64],
    pub dni: Option<&'a [f64]>,
    pub dhi: Option<&'a [f64]>,
    pub air_k: &'a [f64],
    pub wind_ms: &'a [f64],
    pub day: u32,
    pub days_in_year: u32,
    pub utc_seconds: f64,
}
#[derive(Clone, Debug)]
pub struct Cell {
    pub ghi: f32,
    pub dni: f32,
    pub dhi: f32,
    pub clear: f32,
    pub index: f32,
    pub poa: Vec<f32>,
    pub dc: Vec<f32>,
    pub ac: Vec<f32>,
}
pub fn compute(v: Inputs<'_>, systems: &[PvSystem]) -> Result<Vec<Cell>, String> {
    let n = v.lat.len();
    if n == 0
        || v.lon.len() != n
        || v.ghi.len() != n
        || v.air_k.len() != n
        || v.wind_ms.len() != n
        || v.dni.is_some_and(|x| x.len() != n)
        || v.dhi.is_some_and(|x| x.len() != n)
        || v.dni.is_some() != v.dhi.is_some()
    {
        return Err(
            "solar inputs must share one nonempty grid; native DNI/DHI must be paired".into(),
        );
    }
    for p in systems {
        p.validate()?;
    }
    Ok((0..n)
        .into_par_iter()
        .map(|i| {
            let s = sun(
                f64::from(v.lat[i]),
                f64::from(v.lon[i]),
                v.day,
                v.days_in_year,
                v.utc_seconds,
            );
            let good = v.lat[i].is_finite() && v.lon[i].is_finite() && v.lat[i].abs() <= 90.0;
            let ghi = if good { flux(v.ghi[i]) } else { f64::NAN };
            let (dni, dhi) = if let (Some(a), Some(b)) = (v.dni, v.dhi) {
                (flux(a[i]), flux(b[i]))
            } else {
                erbs(ghi, s)
            };
            let clear = if good { clear_sky_ghi(s) } else { f64::NAN };
            let index = if clear >= 20.0 { ghi / clear } else { f64::NAN };
            let mut c = Cell {
                ghi: ghi as f32,
                dni: dni as f32,
                dhi: dhi as f32,
                clear: clear as f32,
                index: index as f32,
                poa: Vec::new(),
                dc: Vec::new(),
                ac: Vec::new(),
            };
            for p in systems {
                let a = poa(p, s, ghi, dni, dhi);
                let (dc, ac) = pvwatts(p, a, v.air_k[i] - 273.15, v.wind_ms[i]);
                c.poa.push(a as f32);
                c.dc.push(dc as f32);
                c.ac.push(ac as f32);
            }
            c
        })
        .collect())
}
/// Signed endpoint ramp MW/min; missing and gapped intervals remain missing.
pub fn ramp(a: &[f32], b: &[f32], seconds: i64, max_gap: i64) -> Result<Vec<f32>, String> {
    if a.len() != b.len() || seconds <= 0 {
        return Err("ramp needs equal grids and increasing times".into());
    }
    Ok(a.par_iter()
        .zip(b)
        .map(|(&a, &b)| {
            if seconds <= max_gap && a.is_finite() && b.is_finite() {
                (b - a) * 60.0 / seconds as f32
            } else {
                f32::NAN
            }
        })
        .collect())
}
/// Trapezoidal energy for a fragment [start,end] of [t0,t1]. Split at UTC midnight.
pub fn integrate(
    acc: &mut [f32],
    a: &[f32],
    b: &[f32],
    t0: i64,
    t1: i64,
    start: i64,
    end: i64,
    max_gap: i64,
    scale: f64,
) -> Result<(), String> {
    if acc.len() != a.len()
        || a.len() != b.len()
        || t1 <= t0
        || start < t0
        || end > t1
        || end <= start
    {
        return Err("invalid solar integration grid or interval".into());
    }
    acc.par_iter_mut().zip(a).zip(b).for_each(|((o, &a), &b)| {
        if t1 - t0 > max_gap || !a.is_finite() || !b.is_finite() {
            *o = f32::NAN;
        } else {
            let fa = (start - t0) as f64 / (t1 - t0) as f64;
            let fb = (end - t0) as f64 / (t1 - t0) as f64;
            let va = f64::from(a) + (f64::from(b) - f64::from(a)) * fa;
            let vb = f64::from(a) + (f64::from(b) - f64::from(a)) * fb;
            *o += ((va + vb) * 0.5 * (end - start) as f64 / 3600.0 * scale) as f32;
        }
    });
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn sys() -> PvSystem {
        serde_json::from_str(include_str!("../data/pv_systems.json"))
            .map(|v: Vec<PvSystem>| v[0].clone())
            .unwrap()
    }
    #[test]
    fn erbs_closure_and_missing() {
        let s = sun(40.0, -88.0, 90, 365, 64800.0);
        for g in [0.0, 50.0, 400.0, 1000.0] {
            let (d, h) = erbs(g, s);
            assert!((d * s.up + h - g).abs() < 1e-8);
        }
        assert!(erbs(f64::NAN, s).0.is_nan());
        assert_eq!(erbs(1.0, Sun { up: -0.2, ..s }), (0.0, 0.0));
    }
    #[test]
    fn tracking_and_backtracking() {
        let mut p = sys();
        p.tracking = true;
        p.tilt_deg = 0.0;
        p.azimuth_deg = 0.0;
        p.max_rotation_deg = 90.0;
        p.backtrack = false;
        let s = Sun {
            east: 0.8,
            north: 0.0,
            up: 0.6,
            extraterrestrial: 1366.0,
        };
        let n = normal(&p, s);
        assert!((n[0] - 0.8).abs() < 1e-10);
        assert!((n[2] - 0.6).abs() < 1e-10);
        p.backtrack = true;
        p.ground_coverage_ratio = 0.8;
        assert!(normal(&p, s)[2] > 0.6);
    }
    #[test]
    fn pvwatts_rating_clipping_missing() {
        let mut p = sys();
        p.losses_fraction = 0.0;
        let (_, a) = pvwatts(&p, 1000.0, 25.0 - 1000.0 / p.thermal_u0, 0.0);
        assert!((a - p.ac_rating_mw()).abs() < 1e-10);
        assert_eq!(pvwatts(&p, 0.0, 20.0, 1.0), (0.0, 0.0));
        assert!(pvwatts(&p, f64::NAN, 20.0, 1.0).1.is_nan());
    }
    #[test]
    fn midnight_gap_ramp() {
        let mut a = [0.0];
        let mut b = [0.0];
        integrate(
            &mut a,
            &[0.0],
            &[2.0],
            82800,
            90000,
            82800,
            86400,
            7200,
            1.0,
        )
        .unwrap();
        integrate(
            &mut b,
            &[0.0],
            &[2.0],
            82800,
            90000,
            86400,
            90000,
            7200,
            1.0,
        )
        .unwrap();
        assert_eq!(a, [0.5]);
        assert_eq!(b, [1.5]);
        assert_eq!(ramp(&[1.0], &[0.0], 300, 600).unwrap(), [-0.2]);
        integrate(&mut a, &[1.0], &[1.0], 0, 7200, 0, 7200, 3600, 1.0).unwrap();
        assert!(a[0].is_nan());
    }
}
