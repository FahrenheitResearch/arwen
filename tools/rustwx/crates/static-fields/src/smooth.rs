//! LANE 2.  geogrid smoothers (`gpuwm/static/build.py`).
//!
//! Per pass: an x sweep then a y sweep, one-cell boundaries untouched
//! in the sweep's own direction; `smth_desmth_special` restores any
//! originally non-negative point made negative -- the WPS terrain
//! smoother (one pass in the build).  Expression order matches the
//! Python exactly: `a + coef * (0.5 * (left + right) - a)`.

use crate::error::{Result, StaticError};
use crate::types::Grid2;

fn one_pass(a: &Grid2, coef: f64) -> Grid2 {
    let (ny, nx) = (a.ny, a.nx);
    // x sweep
    let mut mid = a.clone();
    if nx >= 3 {
        for j in 0..ny {
            for i in 1..nx - 1 {
                let center = a.at(j, i);
                mid.set(
                    j,
                    i,
                    center
                        + coef
                            * (0.5 * (a.at(j, i - 1) + a.at(j, i + 1))
                                - center),
                );
            }
        }
    }
    // y sweep
    let mut out = mid.clone();
    if ny >= 3 {
        for j in 1..ny - 1 {
            for i in 0..nx {
                let center = mid.at(j, i);
                out.set(
                    j,
                    i,
                    center
                        + coef
                            * (0.5 * (mid.at(j - 1, i) + mid.at(j + 1, i))
                                - center),
                );
            }
        }
    }
    out
}

/// 1-2-1 smoother (`one_two_one`).  LANE 2.
pub fn one_two_one(a: &Grid2, passes: usize) -> Result<Grid2> {
    let mut out = a.clone();
    for _ in 0..passes {
        out = one_pass(&out, 0.5);
    }
    Ok(out)
}

/// 0.50 smoothing + (-0.52) desmoothing sweep pair (`smth_desmth`).
pub fn smth_desmth(a: &Grid2, passes: usize) -> Result<Grid2> {
    let mut out = a.clone();
    for _ in 0..passes {
        out = one_pass(&out, 0.5);
        out = one_pass(&out, -0.52);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::{assert_bits_f64, golden_dir, json, read_f64};

    #[test]
    fn smoothers_match_the_python_reference_bitwise() {
        let dir = golden_dir().join("smooth");
        let spec = json(&dir.join("goldens.json"));
        let (dims, data) =
            read_f64(&dir.join(spec["input"].as_str().unwrap()));
        let a = Grid2 {
            ny: dims[0],
            nx: dims[1],
            data,
        };
        for (key, got) in [
            ("one_two_one_1", one_two_one(&a, 1).unwrap()),
            ("one_two_one_2", one_two_one(&a, 2).unwrap()),
            ("smth_desmth_1", smth_desmth(&a, 1).unwrap()),
            ("smth_desmth_special_1", smth_desmth_special(&a, 1).unwrap()),
        ] {
            let (_, want) =
                read_f64(&dir.join(spec[key].as_str().unwrap()));
            assert_bits_f64(&got.data, &want, key);
        }
    }
}

/// WPS terrain smoother with negative-point restoration
/// (`smth_desmth_special`).
pub fn smth_desmth_special(a: &Grid2, passes: usize) -> Result<Grid2> {
    let mut out = smth_desmth(a, passes)?;
    if out.ny != a.ny || out.nx != a.nx {
        return Err(StaticError::Invalid(
            "smoother changed the grid shape".to_string(),
        ));
    }
    for (slot, &orig) in out.data.iter_mut().zip(a.data.iter()) {
        if orig >= 0.0 && *slot < 0.0 {
            *slot = orig;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Per-domain terrain smoothing (gpuwm/static/terrain_smoothing.py).
//
// The default, smth-desmth_special x1, stays the float64 smoother above so
// every default build keeps its bytes; with `"smooth_precision":
// "wps-float32"` it runs the float32 sweeps below instead, which is
// geogrid.exe's own HGT_M.  Every other setting is WPS v4.6.0
// geogrid/src/smooth_module.F in default REAL (f32): per pass an x sweep
// over every row and the interior columns into a scratch plane, then a y
// sweep over the interior rows and columns back into the array, so the
// one-cell outer ring never changes; smth-desmth adds a desmoothing sweep
// pair (1.52, 0.26) after each smoothing pair (0.5, 0.25); the special
// variant restores points made negative from a non-negative start after
// all passes.  Separate f32 multiplies and adds in Fortran's order, never
// mul_add; subnormals are kept.
// ---------------------------------------------------------------------------

/// A GEOGRID.TBL `smooth_option`, or none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
pub enum SmoothOption {
    #[serde(rename = "smth-desmth_special")]
    Special,
    #[serde(rename = "smth-desmth")]
    Desmooth,
    #[serde(rename = "1-2-1")]
    OneTwoOne,
    #[serde(rename = "none")]
    None,
}

/// The arithmetic a setting names (`smooth_precision`).  Only
/// `smth-desmth_special` x1 has a choice: `Float64` (absent from the JSON)
/// is the builders' historical smoother, `WpsFloat32` is WPS's sweeps.
/// Every other smoother runs WPS's sweeps whatever this says, and the
/// Python echo never spells a precision for them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
pub enum SmoothPrecision {
    #[default]
    #[serde(rename = "float64")]
    Float64,
    #[serde(rename = "wps-float32")]
    WpsFloat32,
}

/// One domain's terrain smoothing, as the Python echo writes it:
/// `{"smooth_option": ..., "smooth_passes": ..., "smooth_precision"?: ...}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerrainSmoothing {
    #[serde(rename = "smooth_option")]
    pub option: SmoothOption,
    #[serde(rename = "smooth_passes")]
    pub passes: usize,
    #[serde(rename = "smooth_precision", default)]
    pub precision: SmoothPrecision,
}

/// WPS's stock GEOGRID.TBL setting, the builders' default.
pub const WPS_DEFAULT: TerrainSmoothing = TerrainSmoothing {
    option: SmoothOption::Special,
    passes: 1,
    precision: SmoothPrecision::Float64,
};

impl TerrainSmoothing {
    pub const WPS_DEFAULT: Self = WPS_DEFAULT;

    /// Parse the JSON echo; `none` carries 0 passes, a smoother needs 1+.
    pub fn parse(text: &str) -> Result<Self> {
        let mut value: Self = serde_json::from_str(text).map_err(|e| {
            StaticError::Invalid(format!("terrain smoothing JSON: {e}"))
        })?;
        if value.option == SmoothOption::None
            && value.precision == SmoothPrecision::WpsFloat32
        {
            return Err(StaticError::Invalid(
                "terrain smoothing none runs no smoother; a precision would be ignored"
                    .into(),
            ));
        }
        if value.option == SmoothOption::None {
            value.passes = 0;
        } else if value.passes == 0 {
            return Err(StaticError::Invalid(
                "terrain passes must be positive; use none to disable \
                 smoothing"
                    .into(),
            ));
        }
        Ok(value)
    }
}

/// WPS v4.6.0 smooth_module.F in f32 on one halo-extended plane.
pub fn wps_smooth_f32(a: &Grid2, smoothing: TerrainSmoothing) -> Grid2 {
    let original: Vec<f32> = a.data.iter().map(|v| *v as f32).collect();
    let mut data = original.clone();
    let (ny, nx) = (a.ny, a.nx);
    let sweeps = if smoothing.option == SmoothOption::OneTwoOne { 1 } else { 2 };
    if ny >= 3 && nx >= 3 {
        for _ in 0..smoothing.passes {
            for sweep in 0..sweeps {
                // 0.5*A + 0.25*(pair), then 1.52*A - 0.26*(pair).
                let (c1, c2, desmooth) = if sweep == 0 {
                    (0.5_f32, 0.25_f32, false)
                } else {
                    (1.52_f32, 0.26_f32, true)
                };
                let mut scratch = data.clone();
                for j in 0..ny {
                    for i in 1..nx - 1 {
                        let k = j * nx + i;
                        let t = c1 * data[k];
                        let u = data[k - 1] + data[k + 1];
                        let v = c2 * u;
                        scratch[k] = if desmooth { t - v } else { t + v };
                    }
                }
                for j in 1..ny - 1 {
                    for i in 1..nx - 1 {
                        let k = j * nx + i;
                        let t = c1 * scratch[k];
                        let u = scratch[k - nx] + scratch[k + nx];
                        let v = c2 * u;
                        data[k] = if desmooth { t - v } else { t + v };
                    }
                }
            }
        }
    }
    if smoothing.option == SmoothOption::Special {
        for (slot, orig) in data.iter_mut().zip(original) {
            if *slot < 0.0 && orig >= 0.0 {
                *slot = orig;
            }
        }
    }
    Grid2 {
        ny,
        nx,
        data: data.into_iter().map(f64::from).collect(),
    }
}

/// The builders' terrain smoother for one domain's setting.
pub fn apply_terrain_smoothing(
    a: &Grid2,
    smoothing: TerrainSmoothing,
) -> Result<Grid2> {
    if smoothing == WPS_DEFAULT {
        return smth_desmth_special(a, 1);
    }
    if smoothing.option == SmoothOption::None {
        return Ok(a.clone());
    }
    Ok(wps_smooth_f32(a, smoothing))
}

#[cfg(test)]
mod terrain_smoothing_tests {
    use super::*;
    use crate::testsupport::{assert_bits_f64, golden_dir, json, read_f64};

    /// Replay one golden directory: `goldens.json` names an input plane
    /// (or a `planes` list of them) and, per setting, the expected output.
    /// `oracle_f32` rows are WPS's own output, replayed through the
    /// production entry in WPS's arithmetic: for WPS's default (whose
    /// default path is the legacy float64 smoother) that is
    /// `smooth_precision = "wps-float32"`, and the raw f32 sweeps are held
    /// to the same bytes.
    fn assert_terrain_smoothing_fixture(dir: &std::path::Path) {
        let spec = json(&dir.join("goldens.json"));
        let planes = spec.get("planes").and_then(|v| v.as_array());
        let single = vec![spec.clone()];
        for plane in planes.unwrap_or(&single) {
            let (dims, data) =
                read_f64(&dir.join(plane["input"].as_str().unwrap()));
            let a = Grid2 { ny: dims[0], nx: dims[1], data };
            for row in plane["settings"].as_array().unwrap() {
                let policy = serde_json::json!({
                    "smooth_option": row["smooth_option"],
                    "smooth_passes": row["smooth_passes"],
                });
                let smoothing =
                    TerrainSmoothing::parse(&policy.to_string()).unwrap();
                let oracle = row["oracle_f32"].as_bool().unwrap_or(false);
                let smoothing = if oracle && smoothing == WPS_DEFAULT {
                    TerrainSmoothing {
                        precision: SmoothPrecision::WpsFloat32,
                        ..smoothing
                    }
                } else {
                    smoothing
                };
                let got = apply_terrain_smoothing(&a, smoothing).unwrap();
                let name = row["output"].as_str().unwrap();
                let (_, want) = read_f64(&dir.join(name));
                assert_bits_f64(&got.data, &want, name);
                if oracle && smoothing.option != SmoothOption::None {
                    assert_bits_f64(&wps_smooth_f32(&a, smoothing).data, &want, name);
                }
            }
        }
    }

    #[test]
    fn terrain_smoothing_matches_python_f32_goldens_bitwise() {
        assert_terrain_smoothing_fixture(&golden_dir().join("terrain_smoothing"));
    }

    /// WPS v4.6.0's own output (tools/wps_smooth_v460_oracle/), exported
    /// to GWARR1 by golden/generate_terrain_smoothing_goldens.py.
    #[test]
    fn terrain_smoothing_matches_wps_v460_oracle_bitwise() {
        let dir = std::env::var("GPUWM_WPS_SMOOTH_ORACLE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| golden_dir().join("wps_terrain_smoothing"));
        assert_terrain_smoothing_fixture(&dir);
    }

    #[test]
    fn terrain_smoothing_policy_rejects_unusable_json() {
        for text in [
            r#"{"smooth_option":"1-2-1","smooth_passes":0}"#,
            r#"{"smooth_option":"unknown","smooth_passes":1}"#,
            r#"{"smooth_option":"1-2-1","smooth_passes":true}"#,
            r#"{"smooth_option":"1-2-1","smooth_passes":1,"x":1}"#,
            r#"{"smooth_option":"smth-desmth_special","smooth_passes":1,"smooth_precision":"float32"}"#,
            r#"{"smooth_option":"none","smooth_passes":0,"smooth_precision":"wps-float32"}"#,
        ] {
            assert!(TerrainSmoothing::parse(text).is_err(), "{text}");
        }
    }

    /// The precision key selects WPS's arithmetic for the default smoother
    /// only when it is spelled; without it the default is the legacy
    /// float64 smoother, bit for bit.
    #[test]
    fn terrain_smoothing_precision_selects_wps_arithmetic_for_the_default() {
        let parse = |text: &str| TerrainSmoothing::parse(text).unwrap();
        let default = parse(r#"{"smooth_option":"smth-desmth_special","smooth_passes":1}"#);
        assert_eq!(default, WPS_DEFAULT);
        let exact = parse(
            r#"{"smooth_option":"smth-desmth_special","smooth_passes":1,"smooth_precision":"wps-float32"}"#,
        );
        assert_eq!(exact.precision, SmoothPrecision::WpsFloat32);
        assert_ne!(exact, WPS_DEFAULT);
        let data: Vec<f64> = (0..11 * 13)
            .map(|k| ((k * 7919) % 1601) as f64 - 100.0 + 0.3 * k as f64)
            .collect();
        let a = Grid2 { ny: 11, nx: 13, data };
        let legacy = smth_desmth_special(&a, 1).unwrap();
        let f64_path = apply_terrain_smoothing(&a, default).unwrap();
        let f32_path = apply_terrain_smoothing(&a, exact).unwrap();
        assert_bits_f64(&f64_path.data, &legacy.data, "default");
        assert_bits_f64(&f32_path.data, &wps_smooth_f32(&a, WPS_DEFAULT).data, "exact");
        assert_ne!(f64_path.data, f32_path.data);
    }
}
