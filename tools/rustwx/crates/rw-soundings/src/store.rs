//! rw-store point columns and ensemble column statistics.
//!
//! Two store conventions are read:
//! * `native` (native profile writer): `pressure_native` hPa, `height_native` m,
//!   `temperature_native` K, `dewpoint_native` K, `u_native`/`v_native` m/s
//!   (earth-relative), `omega_native` Pa/s on model levels (level code
//!   1000 - k), plus the 2-D surface anchors.
//! * `iso` (Drew's rw-ui / rusty-weather ingest): `temperature_iso`,
//!   `dewpoint_iso` | `rh_iso`, `u_iso`, `v_iso`, `height_iso` on isobaric
//!   levels, plus the same surface anchors (or their `approx_*` names).
//!
//! The public batch adapter supplies a plain-run inventory with physical valid times.

use rw_store::reader::HourReader;

pub type Res<T> = Result<T, String>;

pub struct Store;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Convention {
    Native,
    Iso,
}

/// Surface anchor (bridge units: hPa, m, C, m/s).
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Sfc {
    #[serde(deserialize_with = "nan_f64")]
    pub psfc_hpa: f64,
    #[serde(deserialize_with = "nan_f64")]
    pub orog_m: f64,
    #[serde(deserialize_with = "nan_f64")]
    pub t2_c: f64,
    #[serde(deserialize_with = "nan_f64")]
    pub td2_c: f64,
    #[serde(deserialize_with = "nan_f64")]
    pub u10: f64,
    #[serde(deserialize_with = "nan_f64")]
    pub v10: f64,
}

/// JSON has no NaN: serde_json writes a missing value as null, and null reads back as NaN.
fn nan_f64<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    Ok(<Option<f64> as serde::Deserialize>::deserialize(d)?.unwrap_or(f64::NAN))
}
fn nan_vec<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
    Ok(<Vec<Option<f64>> as serde::Deserialize>::deserialize(d)?
        .into_iter()
        .map(|x| x.unwrap_or(f64::NAN))
        .collect())
}

/// One raw column, lowest level first, bridge units (hPa, m MSL, C, m/s, Pa/s).
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct RawColumn {
    #[serde(deserialize_with = "nan_vec")]
    pub p: Vec<f64>,
    #[serde(deserialize_with = "nan_vec")]
    pub z: Vec<f64>,
    #[serde(deserialize_with = "nan_vec")]
    pub t: Vec<f64>,
    #[serde(deserialize_with = "nan_vec")]
    pub td: Vec<f64>,
    #[serde(deserialize_with = "nan_vec")]
    pub u: Vec<f64>,
    #[serde(deserialize_with = "nan_vec")]
    pub v: Vec<f64>,
    #[serde(deserialize_with = "nan_vec", default)]
    pub omega: Vec<f64>,
    pub sfc: Sfc,
}

impl RawColumn {
    pub fn levels(&self) -> usize {
        self.p.len()
    }
}

/// Surface anchors, every accepted name (data-lane store first, then the
/// native profile writer / rw-ui names, then rw-ui's approx_* fallbacks).
const SFC_NAMES: [&[&str]; 6] = [
    &["psfc", "surface_pressure", "approx_surface_pressure"],
    &["hgt_sfc", "orography"],
    &["t2m", "temperature_2m", "approx_temperature_2m"],
    &["d2m", "dewpoint_2m", "approx_dewpoint_2m"],
    &["u10", "u_10m", "approx_u_10m"],
    &["v10", "v_10m", "approx_v_10m"],
];

/// Native-level volumes: p, z, T, Td, u, v, omega.
const NATIVE_NAMES: [&[&str]; 7] = [
    &["pres_native", "pressure_native"],
    &["hght_native", "height_native"],
    &["tmp_native", "temperature_native"],
    &["dpt_native", "dewpoint_native"],
    &["u_native"],
    &["v_native"],
    &["omega_native"],
];

fn native_name(reader: &HourReader, i: usize) -> Option<(&'static str, String)> {
    NATIVE_NAMES[i].iter().find_map(|n| {
        reader
            .variable(n)
            .filter(|v| v.kind == "pressure3d")
            .map(|v| (*n, v.units.clone()))
    })
}

/// Native value -> bridge units (hPa, m, C, m/s, Pa/s) by stored units.
fn native_convert(i: usize, v: f64, units: &str) -> f64 {
    match (i, units) {
        (0, "Pa") => v / 100.0,
        (2 | 3, "K") => v - 273.15,
        _ => v,
    }
}

fn sfc_convert(i: usize, v: f64, units: &str) -> f64 {
    match (i, units) {
        (0, "Pa") => v / 100.0,
        (2 | 3, "K") => v - 273.15,
        _ => v,
    }
}

impl Store {
    pub fn convention(reader: &HourReader) -> Res<Convention> {
        if native_name(reader, 0).is_some() {
            Ok(Convention::Native)
        } else if reader.variable("temperature_iso").is_some() {
            Ok(Convention::Iso)
        } else {
            Err("hour has neither native (pressure_native) nor isobaric (temperature_iso) sounding volumes".into())
        }
    }

    fn sfc_name(reader: &HourReader, i: usize) -> Option<(&'static str, String)> {
        for n in SFC_NAMES[i].iter().copied() {
            if let Some(v) = reader.variable(n)
                && v.kind == "surface2d"
            {
                return Some((n, v.units.clone()));
            }
        }
        None
    }

    /// Bilinear point column at fractional grid coords.
    pub fn point_column(reader: &HourReader, fx: f64, fy: f64) -> Res<RawColumn> {
        let conv = Self::convention(reader)?;
        let (nx, ny) = (reader.meta().nx, reader.meta().ny);
        let e = |n: &str, err: rw_store::RwStoreError| format!("read {n}: {err}");
        let prof = |n: &str| -> Res<Vec<f64>> {
            Ok(reader
                .read_profile_3d(n, fx, fy)
                .map_err(|x| e(n, x))?
                .into_iter()
                .map(f64::from)
                .collect())
        };
        // bilinear 2-D
        let x0 = (fx.floor().max(0.0) as usize).min(nx - 1);
        let y0 = (fy.floor().max(0.0) as usize).min(ny - 1);
        let x1 = (x0 + 1).min(nx - 1);
        let y1 = (y0 + 1).min(ny - 1);
        let (ax, ay) = (
            (fx - x0 as f64).clamp(0.0, 1.0),
            (fy - y0 as f64).clamp(0.0, 1.0),
        );
        let pt2 = |n: &str| -> Res<f64> {
            let g = |x, y| {
                reader
                    .read_window_2d(n, x, y, x + 1, y + 1)
                    .map(|w| w.values[0])
                    .map(f64::from)
                    .map_err(|x| e(n, x))
            };
            let (a, b, c, d) = (g(x0, y0)?, g(x1, y0)?, g(x0, y1)?, g(x1, y1)?);
            Ok((a * (1.0 - ax) + b * ax) * (1.0 - ay) + (c * (1.0 - ax) + d * ax) * ay)
        };
        let mut sfc = [f64::NAN; 6];
        for (i, s) in sfc.iter_mut().enumerate() {
            if let Some((n, units)) = Self::sfc_name(reader, i) {
                *s = sfc_convert(i, pt2(n)?, &units);
            }
        }
        let sfc = Sfc {
            psfc_hpa: sfc[0],
            orog_m: sfc[1],
            t2_c: sfc[2],
            td2_c: sfc[3],
            u10: sfc[4],
            v10: sfc[5],
        };
        match conv {
            Convention::Native => {
                let get = |i: usize| -> Res<Vec<f64>> {
                    let (n, u) = native_name(reader, i)
                        .ok_or_else(|| format!("missing native volume {:?}", NATIVE_NAMES[i]))?;
                    Ok(prof(n)?
                        .into_iter()
                        .map(|x| native_convert(i, x, &u))
                        .collect())
                };
                let p = get(0)?;
                let n = p.len();
                let omega = if native_name(reader, 6).is_some() {
                    get(6)?
                } else {
                    vec![f64::NAN; n]
                };
                Ok(RawColumn {
                    p,
                    z: get(1)?,
                    t: get(2)?,
                    td: get(3)?,
                    u: get(4)?,
                    v: get(5)?,
                    omega,
                    sfc,
                })
            }
            Convention::Iso => {
                let levels = reader
                    .variable("temperature_iso")
                    .unwrap()
                    .levels_hpa
                    .clone();
                let units = |n: &str| {
                    reader
                        .variable(n)
                        .map(|v| v.units.clone())
                        .unwrap_or_default()
                };
                let conv_t = |n: &str, v: Vec<f64>| {
                    if units(n) == "K" {
                        v.into_iter().map(|x| x - 273.15).collect()
                    } else {
                        v
                    }
                };
                let t = conv_t("temperature_iso", prof("temperature_iso")?);
                let td = if reader.variable("dewpoint_iso").is_some() {
                    conv_t("dewpoint_iso", prof("dewpoint_iso")?)
                } else {
                    let rh = prof("rh_iso")?;
                    t.iter()
                        .zip(&rh)
                        .map(|(t, rh)| dewpoint_c_from_rh(*t, *rh))
                        .collect()
                };
                Ok(RawColumn {
                    p: levels.iter().map(|l| *l as f64).collect(),
                    z: prof("height_iso")?,
                    t,
                    td,
                    u: prof("u_iso")?,
                    v: prof("v_iso")?,
                    omega: vec![f64::NAN; levels.len()],
                    sfc,
                })
            }
        }
    }

}

/// Same Bolton-style inversion rw-ui uses for RH-only stores.
pub fn dewpoint_c_from_rh(temperature_c: f64, rh_percent: f64) -> f64 {
    let rh = rh_percent.clamp(1.0, 100.0) / 100.0;
    let (a, b) = (17.67, 243.5);
    let gamma = (a * temperature_c) / (b + temperature_c) + rh.ln();
    (b * gamma) / (a - gamma)
}

/// Mean of several member columns, level by level, over one common finite
/// mask per level (the same rule as the area mean).
pub fn mean_columns(cols: &[RawColumn]) -> Res<RawColumn> {
    let first = cols.first().ok_or("no member columns")?;
    let nl = cols.iter().map(|c| c.levels()).min().unwrap_or(0);
    let mut out = RawColumn {
        omega: vec![f64::NAN; nl],
        ..Default::default()
    };
    for k in 0..nl {
        let ok: Vec<&RawColumn> = cols
            .iter()
            .filter(|c| {
                [c.p[k], c.z[k], c.t[k], c.td[k], c.u[k], c.v[k]]
                    .iter()
                    .all(|x| x.is_finite())
            })
            .collect();
        let m = |f: &dyn Fn(&RawColumn) -> f64| {
            if ok.is_empty() {
                f64::NAN
            } else {
                ok.iter().map(|c| f(c)).sum::<f64>() / ok.len() as f64
            }
        };
        out.p.push(m(&|c| c.p[k]));
        out.z.push(m(&|c| c.z[k]));
        out.t.push(m(&|c| c.t[k]));
        out.td.push(m(&|c| c.td[k]));
        out.u.push(m(&|c| c.u[k]));
        out.v.push(m(&|c| c.v[k]));
        let om: Vec<f64> = ok
            .iter()
            .map(|c| c.omega[k])
            .filter(|x| x.is_finite())
            .collect();
        out.omega[k] = if om.is_empty() {
            f64::NAN
        } else {
            om.iter().sum::<f64>() / om.len() as f64
        };
    }
    let sf: Vec<&Sfc> = cols
        .iter()
        .map(|c| &c.sfc)
        .filter(|s| {
            [s.psfc_hpa, s.orog_m, s.t2_c, s.td2_c, s.u10, s.v10]
                .iter()
                .all(|x| x.is_finite())
        })
        .collect();
    let ms = |f: &dyn Fn(&Sfc) -> f64| {
        if sf.is_empty() {
            f64::NAN
        } else {
            sf.iter().map(|s| f(s)).sum::<f64>() / sf.len() as f64
        }
    };
    out.sfc = Sfc {
        psfc_hpa: ms(&|s| s.psfc_hpa),
        orog_m: ms(&|s| s.orog_m),
        t2_c: ms(&|s| s.t2_c),
        td2_c: ms(&|s| s.td2_c),
        u10: ms(&|s| s.u10),
        v10: ms(&|s| s.v10),
    };
    let _ = first;
    Ok(out)
}

/// Linear-interpolated percentile of finite values (q in 0..=100).
pub fn percentile(vals: &mut Vec<f64>, q: f64) -> f64 {
    vals.retain(|x| x.is_finite());
    if vals.is_empty() {
        return f64::NAN;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let pos = (q / 100.0).clamp(0.0, 1.0) * (vals.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    vals[lo] + (vals[hi] - vals[lo]) * (pos - lo as f64)
}

/// p10/p25/p50/p75/p90 per level across members, for T, Td and wind speed.
pub fn percentiles5(cols: &[RawColumn]) -> serde_json::Value {
    let nl = cols.iter().map(|c| c.levels()).min().unwrap_or(0);
    let qs = [10.0, 25.0, 50.0, 75.0, 90.0];
    let names = ["p10", "p25", "p50", "p75", "p90"];
    let mut p = vec![];
    let mut z = vec![];
    let mut out: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; 5]; 3];
    let r = |x: f64| {
        if x.is_finite() {
            serde_json::json!((x * 100.0).round() / 100.0)
        } else {
            serde_json::Value::Null
        }
    };
    for k in 0..nl {
        let mean = |f: &dyn Fn(&RawColumn) -> f64| {
            let v: Vec<f64> = cols.iter().map(f).filter(|x| x.is_finite()).collect();
            if v.is_empty() {
                f64::NAN
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        };
        p.push(r(mean(&|c| c.p[k])));
        z.push(r(mean(&|c| c.z[k])));
        let fields: [Vec<f64>; 3] = [
            cols.iter().map(|c| c.t[k]).collect(),
            cols.iter().map(|c| c.td[k]).collect(),
            cols.iter()
                .map(|c| c.u[k].hypot(c.v[k]) * 1.943_844_5)
                .collect(),
        ];
        for (fi, f) in fields.iter().enumerate() {
            for (qi, q) in qs.iter().enumerate() {
                out[fi][qi].push(percentile(&mut f.clone(), *q));
            }
        }
    }
    let block = |fi: usize| {
        let mut m = serde_json::Map::new();
        for (qi, n) in names.iter().enumerate() {
            m.insert(
                n.to_string(),
                serde_json::Value::Array(out[fi][qi].iter().map(|x| r(*x)).collect()),
            );
        }
        serde_json::Value::Object(m)
    };
    serde_json::json!({"p": p, "z": z, "t": block(0), "td": block(1), "wspd": block(2)})
}

#[cfg(test)]
mod engine_store_tests {
    use super::*;
    #[test]
    fn engine_isobaric_store_profile_units_and_surface_anchors() {
        let path =
            std::env::temp_dir().join(format!("gpuwm-sounding-store-{}.rws", std::process::id()));
        let mut w = rw_store::writer::HourWriter::new(
            "forecast",
            "20260101000000",
            0,
            1,
            1,
            "test",
            "test",
        );
        for (name, unit, values) in [
            ("temperature_iso", "K", [290., 270., 230.]),
            ("dewpoint_iso", "K", [285., 265., 220.]),
            ("height_iso", "m", [800., 3000., 9000.]),
            ("u_iso", "m/s", [5., 15., 25.]),
            ("v_iso", "m/s", [3., 6., 10.]),
        ] {
            let planes: Vec<Vec<f32>> = values.into_iter().map(|v| vec![v]).collect();
            w.add_pressure3d(
                name,
                unit,
                serde_json::json!({}),
                &[900, 700, 300],
                &planes.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
            )
            .unwrap();
        }
        for (name, unit, value) in [
            ("surface_pressure", "Pa", 98000.),
            ("orography", "m", 200.),
            ("temperature_2m", "K", 295.),
            ("dewpoint_2m", "K", 288.),
            ("u_10m", "m/s", 4.),
            ("v_10m", "m/s", 2.),
        ] {
            w.add_surface2d(name, unit, serde_json::json!({}), &[value])
                .unwrap();
        }
        w.finish(&path).unwrap();
        let r = HourReader::open(&path).unwrap();
        let c = Store::point_column(&r, 0., 0.).unwrap();
        assert_eq!(c.p, vec![900., 700., 300.]);
        assert!((c.t[0] - 16.85).abs() < 0.001);
        assert_eq!(c.sfc.psfc_hpa, 980.);
        assert_eq!(c.sfc.u10, 4.);
        assert!((c.sfc.td2_c - 14.85).abs() < 0.001);
        let analyzed = crate::analysis::to_column(&c, Default::default()).unwrap();
        assert_eq!(analyzed.pressure_hpa[0], 980.);
        std::fs::remove_file(path).unwrap();
    }
}
