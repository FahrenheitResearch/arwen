//! Parallel hub-height vectors, isotropic TKE TI estimate and time integrals.
//! Heights outside native levels are missing. The 10 m vector can anchor the
//! layer below the first mass level; no surface TKE is invented.
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Turbine {
    pub id: String,
    pub iec_class: String,
    pub license: String,
    pub source: String,
    pub reference_density_kg_m3: f64,
    pub cut_in_ms: f64,
    pub rated_ms: f64,
    pub cut_out_ms: f64,
    pub points: Vec<[f64; 2]>,
    #[serde(default)]
    pub notes: String,
}
impl Turbine {
    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty()
            || self.source.is_empty()
            || self.license.is_empty()
            || !self.reference_density_kg_m3.is_finite()
            || self.reference_density_kg_m3 <= 0.
            || [self.cut_in_ms, self.rated_ms, self.cut_out_ms]
                .iter()
                .any(|v| !v.is_finite())
            || self.cut_in_ms < 0.
            || self.cut_in_ms >= self.rated_ms
            || self.rated_ms >= self.cut_out_ms
            || self.points.len() < 2
            || self.points.iter().any(|p| {
                !p[0].is_finite() || !p[1].is_finite() || p[0] < 0. || !(0. ..=1.).contains(&p[1])
            })
            || self.points.windows(2).any(|p| p[1][0] <= p[0][0])
            || self.points[0][0] > self.cut_in_ms
            || self.points.last().unwrap()[0] < self.cut_out_ms
        {
            return Err(format!("invalid turbine row {}", self.id));
        }
        Ok(())
    }
    pub fn fraction(&self, s: f64) -> f64 {
        if !s.is_finite() || s < 0. {
            return f64::NAN;
        }
        if s < self.cut_in_ms || s >= self.cut_out_ms {
            return 0.;
        }
        for p in self.points.windows(2) {
            if s >= p[0][0] && s <= p[1][0] {
                return lerp(p[0][1], p[1][1], (s - p[0][0]) / (p[1][0] - p[0][0]));
            }
        }
        f64::NAN
    }
    /// Exact mean of the table curve along a linearly changing wind interval.
    /// Integrate individual segments to preserve the discontinuous cut-out.
    pub fn interval_fraction(&self, a: f64, b: f64) -> f64 {
        if a == b {
            return self.fraction(a);
        }
        let lo = a.min(b);
        let hi = a.max(b);
        let mut knots = vec![lo, hi];
        knots.extend(
            self.points
                .iter()
                .map(|p| p[0])
                .filter(|&x| x > lo && x < hi),
        );
        for x in [self.cut_in_ms, self.cut_out_ms] {
            if x > lo && x < hi {
                knots.push(x);
            }
        }
        knots.sort_by(f64::total_cmp);
        knots.dedup();
        knots
            .windows(2)
            .map(|p| (p[1] - p[0]) * self.fraction((p[0] + p[1]) * 0.5))
            .sum::<f64>()
            / (hi - lo)
    }
}
fn lerp(a: f64, b: f64, f: f64) -> f64 {
    a + (b - a) * f
}
pub struct Columns<'a> {
    pub cells: usize,
    pub levels: usize,
    pub z_agl_m: &'a [f64],
    pub u_ms: &'a [f64],
    pub v_ms: &'a [f64],
    pub sin: &'a [f64],
    pub cos: &'a [f64],
    pub tke: Option<&'a [f64]>,
    pub surface: Option<(&'a [f64], &'a [f64])>,
}
#[derive(Clone)]
pub struct Plane {
    pub height_m: f64,
    pub speed: Vec<f32>,
    pub direction: Vec<f32>,
    pub ti: Vec<f32>,
}
pub fn interpolate(c: Columns<'_>, heights: &[f64]) -> Result<Vec<Plane>, String> {
    validate_columns(&c, heights)?;
    Ok(heights
        .par_iter()
        .map(|&height| {
            let values: Vec<_> = (0..c.cells)
                .into_par_iter()
                .map(|i| {
                    let missing = (f32::NAN, f32::NAN, f32::NAN);
                    let at = |k| k * c.cells + i;
                    if !(0..c.levels)
                        .all(|k| c.z_agl_m[at(k)].is_finite() && c.z_agl_m[at(k)] >= 0.)
                        || (1..c.levels).any(|k| c.z_agl_m[at(k)] <= c.z_agl_m[at(k - 1)])
                        || !c.sin[i].is_finite()
                        || !c.cos[i].is_finite()
                        || (c.sin[i].hypot(c.cos[i]) - 1.).abs() > 0.01
                    {
                        return missing;
                    }
                    let mut pair = None;
                    for k in 0..c.levels {
                        if height == c.z_agl_m[at(k)] {
                            pair = Some((k, k, 0.));
                            break;
                        }
                        if k + 1 < c.levels
                            && height > c.z_agl_m[at(k)]
                            && height < c.z_agl_m[at(k + 1)]
                        {
                            pair = Some((
                                k,
                                k + 1,
                                (height - c.z_agl_m[at(k)])
                                    / (c.z_agl_m[at(k + 1)] - c.z_agl_m[at(k)]),
                            ));
                            break;
                        }
                    }
                    let (u, v, t) = if let Some((a, b, f)) = pair {
                        let ug = lerp(c.u_ms[at(a)], c.u_ms[at(b)], f);
                        let vg = lerp(c.v_ms[at(a)], c.v_ms[at(b)], f);
                        let t = c.tke.map_or(f64::NAN, |t| {
                            if t[at(a)] < 0. || t[at(b)] < 0. {
                                f64::NAN
                            } else {
                                lerp(t[at(a)], t[at(b)], f)
                            }
                        });
                        (
                            ug * c.cos[i] - vg * c.sin[i],
                            ug * c.sin[i] + vg * c.cos[i],
                            t,
                        )
                    } else if height >= 10. && height < c.z_agl_m[i] && c.z_agl_m[i] > 10. {
                        let Some((u, v)) = c.surface else {
                            return missing;
                        };
                        let f = (height - 10.) / (c.z_agl_m[i] - 10.);
                        (
                            lerp(u[i], c.u_ms[i] * c.cos[i] - c.v_ms[i] * c.sin[i], f),
                            lerp(v[i], c.u_ms[i] * c.sin[i] + c.v_ms[i] * c.cos[i], f),
                            f64::NAN,
                        )
                    } else {
                        return missing;
                    };
                    let s = u.hypot(v);
                    if !s.is_finite() {
                        return missing;
                    }
                    let d = if s > 1e-6 {
                        (270. - v.atan2(u).to_degrees()).rem_euclid(360.)
                    } else {
                        f64::NAN
                    };
                    let ti = if t.is_finite() && t >= 0. && s > 1e-6 {
                        (2. * t / 3.).sqrt() / s
                    } else {
                        f64::NAN
                    };
                    (s as f32, d as f32, ti as f32)
                })
                .collect();
            Plane {
                height_m: height,
                speed: values.iter().map(|v| v.0).collect(),
                direction: values.iter().map(|v| v.1).collect(),
                ti: values.iter().map(|v| v.2).collect(),
            }
        })
        .collect())
}
pub fn shear(a: &Plane, b: &Plane) -> Result<Vec<f32>, String> {
    if a.height_m >= b.height_m || a.speed.len() != b.speed.len() {
        return Err("shear needs ordered heights on one grid".into());
    }
    Ok(a.speed
        .par_iter()
        .zip(&b.speed)
        .map(|(&u, &v)| {
            if u > 0. && v > 0. && u.is_finite() && v.is_finite() {
                ((v as f64 / u as f64).ln() / (b.height_m / a.height_m).ln()) as f32
            } else {
                f32::NAN
            }
        })
        .collect())
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub heights_m: Vec<f64>,
    pub turbine: String,
    pub rated_mw: f64,
    pub exceedance_ms: f64,
    pub ramp_window_seconds: f64,
    pub ramp_ms: f64,
    pub power_ramp_fraction: f64,
    pub max_gap_seconds: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            heights_m: vec![80., 100., 120., 150.],
            turbine: "iec_class_2".into(),
            rated_mw: 2.,
            exceedance_ms: 12.,
            ramp_window_seconds: 3600.,
            ramp_ms: 3.,
            power_ramp_fraction: 0.2,
            max_gap_seconds: 5400.,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<(), String> {
        if self.heights_m.is_empty()
            || self.heights_m.iter().any(|h| !h.is_finite() || *h <= 0.)
            || self
                .heights_m
                .iter()
                .enumerate()
                .any(|(i, h)| self.heights_m[..i].contains(h))
            || [
                self.rated_mw,
                self.ramp_window_seconds,
                self.ramp_ms,
                self.power_ramp_fraction,
                self.max_gap_seconds,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.)
            || !self.exceedance_ms.is_finite()
            || self.exceedance_ms < 0.
        {
            return Err("invalid hubwind options".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct Stats {
    pub mean_ms: f32,
    pub max_ms: f32,
    pub exceedance_hours: f32,
    pub cut_in_hours: f32,
    pub rated_hours: f32,
    pub cut_out_hours: f32,
    pub below_cut_in_hours: f32,
    pub valid_hours: f32,
    pub capacity_factor: f32,
    pub energy_mwh: f32,
    pub ramp_up_events: u32,
    pub ramp_down_events: u32,
    pub power_ramp_up_events: u32,
    pub power_ramp_down_events: u32,
    pub max_abs_ramp_ms: f32,
}
fn between(a: f64, b: f64, lo: f64, hi: f64) -> f64 {
    if a == b {
        return if a >= lo && a < hi { 1. } else { 0. };
    }
    ((a.max(b).min(hi) - a.min(b).max(lo)) / (b - a).abs()).clamp(0., 1.)
}
/// Integrate linear speed between timestamps; never bridge missing intervals.
/// Count contiguous signed trailing-window threshold episodes as ramp events.
pub fn statistics(
    times: &[f64],
    speed: &[Vec<f32>],
    curve: &Turbine,
    o: &Options,
) -> Result<Vec<Stats>, String> {
    o.validate()?;
    curve.validate()?;
    let n = speed.first().ok_or("no wind frames")?.len();
    if times.len() != speed.len()
        || times.len() < 2
        || n == 0
        || times.iter().any(|t| !t.is_finite())
        || times.windows(2).any(|t| t[1] <= t[0])
        || speed.iter().any(|s| s.len() != n)
    {
        return Err("statistics need aligned grids and ordered unique times".into());
    }
    Ok((0..n)
        .into_par_iter()
        .map(|c| {
            let mut r = Stats {
                mean_ms: f32::NAN,
                max_ms: f32::NAN,
                exceedance_hours: 0.,
                cut_in_hours: 0.,
                rated_hours: 0.,
                cut_out_hours: 0.,
                below_cut_in_hours: 0.,
                valid_hours: 0.,
                capacity_factor: f32::NAN,
                energy_mwh: 0.,
                ramp_up_events: 0,
                ramp_down_events: 0,
                power_ramp_up_events: 0,
                power_ramp_down_events: 0,
                max_abs_ramp_ms: f32::NAN,
            };
            let (mut sum, mut energy, mut hours) = (0., 0., 0.);
            let mut maximum = f64::NAN;
            let (mut last, mut lastp, mut start) = (0, 0, 0);
            for j in 0..times.len() {
                let b = speed[j][c] as f64;
                if b.is_finite() && b >= 0. {
                    maximum = if maximum.is_nan() { b } else { maximum.max(b) };
                }
                if j == 0 {
                    continue;
                }
                let a = speed[j - 1][c] as f64;
                let dt = times[j] - times[j - 1];
                if !a.is_finite() || !b.is_finite() || a < 0. || b < 0. || dt > o.max_gap_seconds {
                    last = 0;
                    lastp = 0;
                    start = j;
                    continue;
                }
                let h = dt / 3600.;
                hours += h;
                sum += h * (a + b) * 0.5;
                energy += h * curve.interval_fraction(a, b);
                r.exceedance_hours += (h * between(a, b, o.exceedance_ms, f64::INFINITY)) as f32;
                r.below_cut_in_hours += (h * between(a, b, 0., curve.cut_in_ms)) as f32;
                r.cut_in_hours += (h * between(a, b, curve.cut_in_ms, curve.rated_ms)) as f32;
                r.rated_hours += (h * between(a, b, curve.rated_ms, curve.cut_out_ms)) as f32;
                r.cut_out_hours += (h * between(a, b, curve.cut_out_ms, f64::INFINITY)) as f32;
                let target = times[j] - o.ramp_window_seconds;
                if target < times[start] {
                    last = 0;
                    lastp = 0;
                    continue;
                }
                let k = (times.partition_point(|&t| t <= target).max(1) - 1).min(times.len() - 2);
                let old = lerp(
                    speed[k][c] as f64,
                    speed[k + 1][c] as f64,
                    (target - times[k]) / (times[k + 1] - times[k]),
                );
                let d = b - old;
                let dp = curve.fraction(b) - curve.fraction(old);
                r.max_abs_ramp_ms = if r.max_abs_ramp_ms.is_nan() {
                    d.abs() as f32
                } else {
                    r.max_abs_ramp_ms.max(d.abs() as f32)
                };
                let now = if d >= o.ramp_ms {
                    1
                } else if d <= -o.ramp_ms {
                    -1
                } else {
                    0
                };
                let nowp = if dp >= o.power_ramp_fraction {
                    1
                } else if dp <= -o.power_ramp_fraction {
                    -1
                } else {
                    0
                };
                if now != last {
                    if now == 1 {
                        r.ramp_up_events += 1;
                    }
                    if now == -1 {
                        r.ramp_down_events += 1;
                    }
                }
                if nowp != lastp {
                    if nowp == 1 {
                        r.power_ramp_up_events += 1;
                    }
                    if nowp == -1 {
                        r.power_ramp_down_events += 1;
                    }
                }
                last = now;
                lastp = nowp;
            }
            r.max_ms = maximum as f32;
            r.valid_hours = hours as f32;
            if hours > 0. {
                r.mean_ms = (sum / hours) as f32;
                r.capacity_factor = (energy / hours) as f32;
                r.energy_mwh = (energy * o.rated_mw) as f32;
            } else {
                r.energy_mwh = f32::NAN;
                r.exceedance_hours = f32::NAN;
                r.cut_in_hours = f32::NAN;
                r.rated_hours = f32::NAN;
                r.cut_out_hours = f32::NAN;
                r.below_cut_in_hours = f32::NAN;
            }
            r
        })
        .collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn curve() -> Turbine {
        Turbine {
            id: "test".into(),
            iec_class: "test".into(),
            license: "Apache-2.0".into(),
            source: "test".into(),
            reference_density_kg_m3: 1.225,
            cut_in_ms: 3.,
            rated_ms: 10.,
            cut_out_ms: 25.,
            points: vec![[0., 0.], [3., 0.], [10., 1.], [25., 1.]],
            notes: String::new(),
        }
    }
    #[test]
    fn hard_cutout() {
        let c = curve();
        c.validate().unwrap();
        assert_eq!(c.fraction(25.), 0.);
        assert!((c.interval_fraction(24., 26.) - 0.5).abs() < 1e-12);
    }
    #[test]
    fn vector_rotation_wrap_and_missing() {
        let p = interpolate(
            Columns {
                cells: 1,
                levels: 2,
                z_agl_m: &[50., 150.],
                u_ms: &[10., 10.],
                v_ms: &[-1., 1.],
                sin: &[0.],
                cos: &[1.],
                tke: None,
                surface: None,
            },
            &[100., 200.],
        )
        .unwrap();
        assert_eq!(p[0].speed[0], 10.);
        assert_eq!(p[0].direction[0], 270.);
        assert!(p[0].ti[0].is_nan());
        assert!(p[1].speed[0].is_nan());
    }
    #[test]
    fn clock_coverage_and_ramps() {
        let o = Options {
            max_gap_seconds: 4000.,
            ..Default::default()
        };
        let r = statistics(
            &[0., 1800., 3600., 10800.],
            &[vec![0.], vec![6.], vec![12.], vec![24.]],
            &curve(),
            &o,
        )
        .unwrap();
        assert_eq!(r[0].mean_ms, 6.);
        assert_eq!(r[0].valid_hours, 1.);
        assert_eq!(r[0].ramp_up_events, 1);
        assert_eq!(r[0].max_abs_ramp_ms, 12.);
        assert!((r[0].below_cut_in_hours + r[0].cut_in_hours + r[0].rated_hours - 1.).abs() < 1e-6);
    }
    #[test]
    fn ramp_window_starts_exactly_after_missing_sample() {
        let o = Options {
            max_gap_seconds: 2000.,
            ..Default::default()
        };
        let r = statistics(
            &[0., 1800., 3600., 5400.],
            &[vec![f32::NAN], vec![2.], vec![8.], vec![14.]],
            &curve(),
            &o,
        )
        .unwrap();
        assert_eq!(r[0].valid_hours, 1.);
        assert_eq!(r[0].mean_ms, 8.);
        assert_eq!(r[0].max_abs_ramp_ms, 12.);
        assert_eq!(r[0].ramp_up_events, 1);
    }
    #[test]
    fn absent_coverage_never_becomes_zero_capacity_factor() {
        let r = statistics(
            &[0., 7200.],
            &[vec![10.], vec![10.]],
            &curve(),
            &Options::default(),
        )
        .unwrap();
        assert_eq!(r[0].valid_hours, 0.);
        assert!(r[0].capacity_factor.is_nan());
        assert!(r[0].energy_mwh.is_nan());
        assert!(statistics(
            &[0., 0.],
            &[vec![10.], vec![10.]],
            &curve(),
            &Options::default()
        )
        .is_err());
    }
    #[test]
    fn ti_and_shear() {
        let p = interpolate(
            Columns {
                cells: 1,
                levels: 2,
                z_agl_m: &[50., 200.],
                u_ms: &[5., 10.],
                v_ms: &[0., 0.],
                sin: &[0.],
                cos: &[1.],
                tke: Some(&[1.5, 1.5]),
                surface: None,
            },
            &[50., 200.],
        )
        .unwrap();
        assert!((p[0].ti[0] - 0.2).abs() < 1e-6);
        assert!((shear(&p[0], &p[1]).unwrap()[0] - 0.5).abs() < 1e-6);
    }
}

pub(crate) fn validate_columns(c: &Columns<'_>, heights: &[f64]) -> Result<(), String> {
    let n = c
        .cells
        .checked_mul(c.levels)
        .ok_or("column size overflow")?;
    if c.cells == 0
        || c.levels == 0
        || [c.z_agl_m.len(), c.u_ms.len(), c.v_ms.len()]
            .iter()
            .any(|&v| v != n)
        || c.sin.len() != c.cells
        || c.cos.len() != c.cells
        || c.tke.is_some_and(|v| v.len() != n)
        || c.surface
            .is_some_and(|(u, v)| u.len() != c.cells || v.len() != c.cells)
        || heights.is_empty()
        || heights.iter().any(|h| !h.is_finite() || *h <= 0.)
        || heights
            .iter()
            .enumerate()
            .any(|(i, h)| heights[..i].contains(h))
    {
        return Err("invalid hubwind columns, rotation or height list".into());
    }
    Ok(())
}
