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
