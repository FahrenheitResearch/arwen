//! The case-7 vertical interpolation kernel.
//!
//! A port of `vertical_interp` in `mpas_init_atm_vinterp.F`: linear in the
//! vertical coordinate, with three extrapolation modes below the column and
//! two above it.
//!
//! ## The `order` argument
//! The Fortran takes an `order` argument, defaults it to 2, and then never
//! reads it.  Every case-7 call site passes `order = 1`.  It is dropped here
//! rather than carried as a dead parameter, and this paragraph is the record
//! of that: there is no quadratic path in the Fortran to port, and inventing
//! one would silently change every profile.
//!
//! ## The 99999.0 surface sentinel
//! Callers replace the surface level's coordinate with `99999.0` before
//! sorting, which floats that level to the top of the ascending column, and
//! then hand this function only `1 ..= n-1` of the sorted column.  The net
//! effect is that **the surface level is excluded from every profile
//! interpolation** — deliberately, and only surface pressure opts out by
//! passing the full column.  See [`SURFACE_SENTINEL`] and
//! [`sorted_column`].

#![allow(clippy::needless_range_loop)]

/// The `vert_level` tag that marks the surface level in an intermediate file.
pub const SURFACE_LEVEL_TAG: f32 = 200100.0;

/// The coordinate the surface level is given before sorting, so that it sorts
/// to the top of the column and the caller's `1 ..= n-1` slice drops it.
pub const SURFACE_SENTINEL: f32 = 99999.0;

/// Behaviour outside the column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extrap {
    /// 0: hold the end value.
    Constant,
    /// 1: continue the slope of the two end levels.
    Linear,
    /// 2: a 6.5 K/km lapse rate below the column; **not implemented above it**
    /// in the Fortran, where it is a fatal error.  Reproduced as a refusal.
    LapseRate,
}

impl Extrap {
    /// Parse `config_extrap_airtemp`.  There is no default: the three modes
    /// give visibly different low-level temperatures over high terrain, and a
    /// wrong guess is a plausible-looking file.
    pub fn parse(text: &str) -> Result<Extrap, String> {
        match text {
            "constant" => Ok(Extrap::Constant),
            "linear" => Ok(Extrap::Linear),
            "lapse-rate" => Ok(Extrap::LapseRate),
            other => Err(format!(
                "unknown extrapolation mode \"{other}\"; config_extrap_airtemp takes \
                 constant, linear or lapse-rate"
            )),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Extrap::Constant => "constant",
            Extrap::Linear => "linear",
            Extrap::LapseRate => "lapse-rate",
        }
    }
}

/// Raised when the column is asked to extrapolate above its top under the
/// lapse-rate mode, which `mpas_init_atm_vinterp.F` leaves unimplemented and
/// turns into a fatal error at the call site.
#[derive(Debug, Clone, Copy)]
pub struct AboveTopLapseRate;

/// A column of `(coordinate, value)` pairs, sorted ascending by coordinate.
///
/// Sorting is stable here where `mpas_quicksort` is not.  On a first-guess
/// column the coordinates are distinct geopotential heights, so the two agree;
/// where they could not — two levels reported at exactly the same height — a
/// stable order is at least reproducible run to run, which an unstable
/// quicksort's is not.
pub fn sorted_column(coords: &mut [f32], values: &mut [f32]) {
    let n = coords.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        coords[a]
            .partial_cmp(&coords[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let c: Vec<f32> = idx.iter().map(|&i| coords[i]).collect();
    let v: Vec<f32> = idx.iter().map(|&i| values[i]).collect();
    coords.copy_from_slice(&c);
    values.copy_from_slice(&v);
}

/// Interpolate `values` (paired with ascending `coords`) to `target`.
pub fn vertical_interp(
    target: f32,
    coords: &[f32],
    values: &[f32],
    extrap: Extrap,
) -> Result<f32, AboveTopLapseRate> {
    let nz = coords.len();
    debug_assert_eq!(nz, values.len());
    debug_assert!(nz >= 2);

    if target < coords[0] {
        return Ok(match extrap {
            Extrap::Constant => values[0],
            Extrap::Linear => {
                let slope = (values[1] - values[0]) / (coords[1] - coords[0]);
                values[0] + slope * (target - coords[0])
            }
            Extrap::LapseRate => values[0] - (target - coords[0]) * 0.0065,
        });
    }
    if target >= coords[nz - 1] {
        return match extrap {
            Extrap::Constant => Ok(values[nz - 1]),
            Extrap::Linear => {
                let slope =
                    (values[nz - 1] - values[nz - 2]) / (coords[nz - 1] - coords[nz - 2]);
                Ok(values[nz - 1] + slope * (target - coords[nz - 1]))
            }
            Extrap::LapseRate => Err(AboveTopLapseRate),
        };
    }

    for k in 0..nz - 1 {
        if target >= coords[k] && target < coords[k + 1] {
            let wm = (coords[k + 1] - target) / (coords[k + 1] - coords[k]);
            let wp = (target - coords[k]) / (coords[k + 1] - coords[k]);
            return Ok(wm * values[k] + wp * values[k + 1]);
        }
    }
    // Unreachable for a sorted column with the two guards above; the Fortran
    // would use whatever `wm`/`wp` were left from a previous call here.
    Ok(values[nz - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inside_the_column_it_is_linear_in_the_coordinate() {
        let z = [0.0f32, 100.0];
        let f = [10.0f32, 20.0];
        let v = vertical_interp(25.0, &z, &f, Extrap::Linear).unwrap();
        assert!((v - 12.5).abs() < 1e-5, "{v}");
    }

    #[test]
    fn below_the_column_each_mode_gives_its_own_answer() {
        let z = [100.0f32, 200.0];
        let f = [10.0f32, 20.0];
        assert_eq!(
            vertical_interp(0.0, &z, &f, Extrap::Constant).unwrap(),
            10.0
        );
        let lin = vertical_interp(0.0, &z, &f, Extrap::Linear).unwrap();
        assert!((lin - 0.0).abs() < 1e-4, "{lin}");
        let lapse = vertical_interp(0.0, &z, &f, Extrap::LapseRate).unwrap();
        assert!((lapse - (10.0 + 100.0 * 0.0065)).abs() < 1e-5, "{lapse}");
    }

    #[test]
    fn lapse_rate_above_the_top_is_the_fortran_fatal() {
        let z = [100.0f32, 200.0];
        let f = [10.0f32, 20.0];
        assert!(vertical_interp(500.0, &z, &f, Extrap::LapseRate).is_err());
    }

    #[test]
    fn the_surface_sentinel_sorts_the_surface_level_out_of_the_profile() {
        // Three levels: two aloft plus a surface level whose coordinate has
        // been replaced by the sentinel.  After sorting, the caller's
        // 1..=n-1 slice must be the two real levels.
        let mut coords = [500.0f32, SURFACE_SENTINEL, 1500.0];
        let mut values = [1.0f32, 99.0, 3.0];
        sorted_column(&mut coords, &mut values);
        assert_eq!(coords[2], SURFACE_SENTINEL);
        let profile_c = &coords[..2];
        let profile_v = &values[..2];
        assert_eq!(profile_c, &[500.0, 1500.0]);
        assert_eq!(profile_v, &[1.0, 3.0]);
    }

    #[test]
    fn an_unknown_extrapolation_mode_is_refused_by_name() {
        let err = Extrap::parse("linear-ish").unwrap_err();
        assert!(err.contains("linear-ish"), "{err}");
        assert!(err.contains("lapse-rate"), "{err}");
    }
}
