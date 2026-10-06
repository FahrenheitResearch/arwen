//! The height of a pressure surface, read where a WRF-family model knows its
//! height: at the INTERFACES of its layers.
//!
//! A WRF-family model carries geopotential (PH + PHB) on the interfaces
//! between its layers and pressure (P + PB) at the layers' middles.  The
//! usual chart product first averages the two interface heights of each
//! layer and calls that the height at the layer's pressure.  It is not: on
//! an eta grid the layer's pressure is the arithmetic mean of its two
//! interface pressures, and the mean of two heights is the height at their
//! GEOMETRIC mean pressure, which is lower.  The averaged height is
//! therefore too high for the pressure it is paired with: 4 to 6 m at
//! 500 hPa on a 50-level grid, nothing on a chart and the whole signal on a
//! difference panel against a post-processor that does not make the same
//! pairing.
//!
//! This reads the surface between interfaces instead.  Each interface's
//! pressure is found from the two mass-level pressures around it, linearly
//! in the model's vertical coordinate (in which pressure is very nearly
//! linear), and the height is interpolated between interfaces in
//! log-pressure.  A column whose surface pressure is below the level has no
//! such surface and is left NaN rather than extrapolated underground.
//!
//! Every consumer shares this one implementation: the renderer's isobaric
//! chart planes and sounding volumes (`rw-wrfbatch`), the machine-learning
//! exporter's geopotential on pressure levels (`rw-mlexport`), and through
//! the C ABI in `capi.rs` the Python consumers (the moving-nest vortex
//! tracker on the host, the GNSS-RO refractivity operator, the verification
//! diagnostics and the flagship products).  The tracker's device path is a
//! CUDA transcription of [`column_isobaric_height`] graded bit for bit
//! against it.  The comparison sheets (`rw_compare`) measured the method
//! first: the plane read this way sits 0.001 dam from the reference model's
//! own 500 hPa analysis over its whole grid.
//!
//! THE LOGARITHM IS `libm::log`, not `f64::ln`.  `f64::ln` is the platform
//! C library, whose last bit differs between glibc releases and the
//! Windows runtime; `libm` 0.2.16 is pure IEEE arithmetic, the same bits on
//! every host, and `gpuwm/core/kernels/portable_libm64.cuh` transcribes it
//! for the card (`plm_log`), which is what lets the device path equal this
//! one word for word.

/// Standard gravity (m s-2).  Geopotential over this is geopotential
/// metres, which is what a GRIB height message is and what every other
/// height in the store is.
pub const STANDARD_GRAVITY: f64 = 9.80665;

/// The natural logarithm every height here is read with (see the module
/// note): Rust `libm` 0.2.16, transcribed for the card as `plm_log`.
#[inline]
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}

/// A column's interface heights: `values` (plus `plus`, elementwise, when
/// the model splits its geopotential into a perturbation and a base state,
/// WRF's PH and PHB) over `per_metre`, `[(levels + 1) * cells]` bottom up
/// with the level index outermost.
#[derive(Debug, Clone, Copy)]
pub struct Interfaces<'a> {
    pub values: &'a [f64],
    pub plus: Option<&'a [f64]>,
    pub per_metre: f64,
}

impl<'a> Interfaces<'a> {
    /// Heights in metres, or geopotential with its gravity.
    pub fn new(values: &'a [f64], per_metre: f64) -> Self {
        Self {
            values,
            plus: None,
            per_metre,
        }
    }

    /// `values + plus` over `per_metre`: WRF's PH + PHB in one read.
    pub fn split(values: &'a [f64], plus: &'a [f64], per_metre: f64) -> Self {
        Self {
            values,
            plus: Some(plus),
            per_metre,
        }
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn check(&self, nz: usize, cells: usize) -> Result<(), String> {
        let expected = (nz + 1) * cells;
        if self.values.len() != expected || self.plus.is_some_and(|plus| plus.len() != expected) {
            return Err(format!(
                "isobaric height needs {} x {cells} interface values; got {}{}",
                nz + 1,
                self.values.len(),
                self.plus
                    .map(|plus| format!(" and {}", plus.len()))
                    .unwrap_or_default()
            ));
        }
        if !self.per_metre.is_finite() || self.per_metre <= 0.0 {
            return Err(format!("{} interface units per metre is not a scale", self.per_metre));
        }
        Ok(())
    }

    /// The height (m) of interface `index` (`k * cells + cell`).
    #[inline]
    pub fn metres(&self, index: usize) -> f64 {
        let value = match self.plus {
            Some(plus) => self.values[index] + plus[index],
            None => self.values[index],
        };
        value / self.per_metre
    }
}

/// Where each layer interface of a column sits between the two mass levels
/// around it, in the model's vertical coordinate.
///
/// The same for every column of a grid: the vertical coordinate (WRF's
/// `ZNU` and `ZNW`) is one profile.  The bottom and top interfaces lie
/// outside the mass levels and are reached by the same line continued.
#[derive(Debug, Clone, PartialEq)]
pub struct InterfaceStencil {
    /// The lower of the two mass levels each interface is read from.
    lower: Vec<usize>,
    /// The interface's position from that level toward the next, in the
    /// vertical coordinate (outside 0..1 at the bottom and top).
    weight: Vec<f64>,
}

impl InterfaceStencil {
    /// The stencil of a grid whose mass levels sit at `eta_mass` and whose
    /// interfaces sit at `eta_interface` (one more interface than levels).
    pub fn new(eta_mass: &[f64], eta_interface: &[f64]) -> Result<Self, String> {
        let levels = eta_mass.len();
        if levels < 2 || eta_interface.len() != levels + 1 {
            return Err(format!(
                "{levels} mass level(s) and {} interface(s) are not one more interface than levels",
                eta_interface.len()
            ));
        }
        if eta_mass
            .iter()
            .chain(eta_interface)
            .any(|value| !value.is_finite())
        {
            return Err("the vertical coordinate holds a value that is not a number".into());
        }
        let mut lower = vec![0usize; levels + 1];
        let mut weight = vec![0.0f64; levels + 1];
        for interface in 0..=levels {
            let below = interface.saturating_sub(1).min(levels - 2);
            let span = eta_mass[below + 1] - eta_mass[below];
            if !span.is_finite() || span == 0.0 {
                return Err("two mass levels share one vertical coordinate".into());
            }
            lower[interface] = below;
            weight[interface] = (eta_interface[interface] - eta_mass[below]) / span;
        }
        Ok(Self { lower, weight })
    }

    /// The stencil of a grid that states only its interfaces, with each mass
    /// level at the middle of its layer in the vertical coordinate -- how WRF
    /// defines `ZNU` from `ZNW`.
    pub fn from_interfaces(eta_interface: &[f64]) -> Result<Self, String> {
        let eta_mass: Vec<f64> = eta_interface
            .windows(2)
            .map(|pair| 0.5 * (pair[0] + pair[1]))
            .collect();
        Self::new(&eta_mass, eta_interface)
    }

    /// The stencil of a grid that states its layer thicknesses in the
    /// vertical coordinate (WRF's `DNW`, `ZNW[k + 1] - ZNW[k]`, negative
    /// upward) with the ground at 1: the interfaces are their running sum.
    pub fn from_layer_thickness(dnw: &[f64]) -> Result<Self, String> {
        Self::from_interfaces(&interfaces_from_layer_thickness(dnw))
    }

    /// Mass levels per column.
    pub fn levels(&self) -> usize {
        self.lower.len() - 1
    }

    /// The natural log of every interface pressure of one column, bottom up,
    /// from the column's mass-level pressures (`pressure(k)` for mass level
    /// `k`, any one unit).  `false` when an interface pressure is not a
    /// positive number: the column has no usable interfaces, and `ln_p` is
    /// then only partly written.
    pub fn interface_ln_pressures(
        &self,
        pressure: impl Fn(usize) -> f64,
        ln_p: &mut [f64],
    ) -> bool {
        debug_assert_eq!(ln_p.len(), self.lower.len());
        for (interface, slot) in ln_p.iter_mut().enumerate() {
            let below = self.lower[interface];
            let (p_lower, p_upper) = (pressure(below), pressure(below + 1));
            let value = p_lower + (p_upper - p_lower) * self.weight[interface];
            if !value.is_finite() || value <= 0.0 {
                return false;
            }
            *slot = ln(value);
        }
        true
    }
}

/// The interfaces of a coordinate stated as layer thicknesses (`DNW`), from
/// the ground at 1: `ZNW[k + 1] = ZNW[k] + DNW[k]`.
pub fn interfaces_from_layer_thickness(dnw: &[f64]) -> Vec<f64> {
    let mut eta = Vec::with_capacity(dnw.len() + 1);
    let mut value = 1.0f64;
    eta.push(value);
    for thickness in dnw {
        value += thickness;
        eta.push(value);
    }
    eta
}

/// The height of the surface `target_ln_p` (ln of its pressure) in one
/// column whose interface log-pressures are `ln_p`: `None` where no layer
/// straddles it or an interface height is not a number.
///
/// THE ONE COLUMN READ every consumer runs, and the routine the card's
/// transcription (the `isobaric_height` kernel in
/// `gpuwm/core/storm_tracking.py`) is graded against word for word: the
/// bracket scans from the ground, the fraction is linear in ln p, and the
/// height is linear in that fraction.
#[inline]
pub fn column_isobaric_height(
    ln_p: &[f64],
    target_ln_p: f64,
    interfaces: &Interfaces<'_>,
    cells: usize,
    cell: usize,
) -> Option<f64> {
    let (below, fraction) = interface_bracket(ln_p, target_ln_p)?;
    let z_lower = interfaces.metres(below * cells + cell);
    let z_upper = interfaces.metres((below + 1) * cells + cell);
    (z_lower.is_finite() && z_upper.is_finite())
        .then(|| z_lower + (z_upper - z_lower) * fraction)
}

/// The interface below `target_ln_p` in a column and the log-pressure
/// fraction of the way to the interface above, scanning from the ground up.
/// `None` when no layer of the column straddles the target: it lies under
/// the ground or above the model top.
pub fn interface_bracket(ln_p: &[f64], target_ln_p: f64) -> Option<(usize, f64)> {
    (0..ln_p.len().saturating_sub(1)).find_map(|interface| {
        let (lower, upper) = (ln_p[interface], ln_p[interface + 1]);
        (lower >= target_ln_p && upper < target_ln_p)
            .then(|| (interface, (target_ln_p - lower) / (upper - lower)))
    })
}

/// The height of every isobaric surface in `levels` in every column, read
/// between layer interfaces.
///
/// `interface` is `[(levels + 1) * cells]` interface values, bottom up with
/// the level index outermost, in any unit of which `per_metre` makes one
/// metre: geopotential (m2 s-2) with [`STANDARD_GRAVITY`] for geopotential
/// metres, or heights with `1.0`.  `p_mass` is `[levels * cells]` mass-level
/// pressures and `levels` the target pressures, in one unit.  Returns one
/// plane of `cells` per target, NaN where a column has no such surface.
pub fn isobaric_heights_from_interfaces(
    interface: &[f64],
    per_metre: f64,
    p_mass: &[f64],
    stencil: &InterfaceStencil,
    cells: usize,
    levels: &[f64],
) -> Result<Vec<Vec<f32>>, String> {
    let mut planes = vec![vec![f32::NAN; cells]; levels.len()];
    isobaric_heights_into(interface, per_metre, p_mass, stencil, cells, levels, &mut planes)?;
    Ok(planes)
}

/// [`isobaric_heights_from_interfaces`] into planes the caller owns, one of
/// `cells` per target level, for a caller that reserves its output buffers
/// itself.  Every value of every plane is written, NaN where the column has
/// no such surface.
pub fn isobaric_heights_into(
    interface: &[f64],
    per_metre: f64,
    p_mass: &[f64],
    stencil: &InterfaceStencil,
    cells: usize,
    levels: &[f64],
    planes: &mut [Vec<f32>],
) -> Result<(), String> {
    if planes.len() != levels.len() || planes.iter().any(|plane| plane.len() != cells) {
        return Err(format!(
            "isobaric height writes {} plane(s) of {cells}; was given {} plane(s)",
            levels.len(),
            planes.len()
        ));
    }
    for_each_isobaric_height(
        &Interfaces::new(interface, per_metre),
        p_mass,
        stencil,
        cells,
        levels,
        |level, cell, value| planes[level][cell] = value.map_or(f32::NAN, |v| v as f32),
    )
}

/// Every isobaric height as f64, `[levels.len() * cells]` level-major, NaN
/// where a column has no such surface: the full-precision answer the C ABI
/// hands the Python consumers and the card's transcription is graded on.
pub fn isobaric_heights_f64_into(
    interfaces: &Interfaces<'_>,
    p_mass: &[f64],
    stencil: &InterfaceStencil,
    cells: usize,
    levels: &[f64],
    out: &mut [f64],
) -> Result<(), String> {
    if out.len() != levels.len() * cells {
        return Err(format!(
            "isobaric height writes {} x {cells} values; was given room for {}",
            levels.len(),
            out.len()
        ));
    }
    for_each_isobaric_height(interfaces, p_mass, stencil, cells, levels, |level, cell, value| {
        out[level * cells + cell] = value.unwrap_or(f64::NAN)
    })
}

fn check_columns(
    interfaces: &Interfaces<'_>,
    p_mass: &[f64],
    stencil: &InterfaceStencil,
    cells: usize,
) -> Result<usize, String> {
    let nz = stencil.levels();
    if cells == 0 || p_mass.len() != nz * cells {
        return Err(format!(
            "isobaric height needs {nz} x {cells} pressures and {} x {cells} interface heights; got {} and {}",
            nz + 1,
            p_mass.len(),
            interfaces.len()
        ));
    }
    interfaces.check(nz, cells)?;
    Ok(nz)
}

fn for_each_isobaric_height(
    interfaces: &Interfaces<'_>,
    p_mass: &[f64],
    stencil: &InterfaceStencil,
    cells: usize,
    levels: &[f64],
    mut write: impl FnMut(usize, usize, Option<f64>),
) -> Result<(), String> {
    let nz = check_columns(interfaces, p_mass, stencil, cells)?;
    if let Some(bad) = levels.iter().find(|level| !level.is_finite() || **level <= 0.0) {
        return Err(format!("{bad} is not a pressure level"));
    }
    let targets: Vec<f64> = levels.iter().map(|level| ln(*level)).collect();
    let mut ln_p = vec![0.0f64; nz + 1];
    for cell in 0..cells {
        let usable = stencil.interface_ln_pressures(|k| p_mass[k * cells + cell], &mut ln_p);
        for (level, &target) in targets.iter().enumerate() {
            let value = if usable {
                column_isobaric_height(&ln_p, target, interfaces, cells, cell)
            } else {
                None
            };
            write(level, cell, value);
        }
    }
    Ok(())
}

/// ln p on every interface, `[(levels + 1) * cells]`, from the mass-level
/// pressures; a column with an interface pressure that is not a positive
/// number is NaN throughout.
pub fn interface_log_pressures_into(
    p_mass: &[f64],
    stencil: &InterfaceStencil,
    cells: usize,
    out: &mut [f64],
) -> Result<(), String> {
    let nz = stencil.levels();
    if cells == 0 || p_mass.len() != nz * cells || out.len() != (nz + 1) * cells {
        return Err(format!(
            "interface log-pressure needs {nz} x {cells} pressures and room for {} x {cells}; got {} and {}",
            nz + 1,
            p_mass.len(),
            out.len()
        ));
    }
    let mut ln_p = vec![0.0f64; nz + 1];
    for cell in 0..cells {
        let usable = stencil.interface_ln_pressures(|k| p_mass[k * cells + cell], &mut ln_p);
        for (k, value) in ln_p.iter().enumerate() {
            out[k * cells + cell] = if usable { *value } else { f64::NAN };
        }
    }
    Ok(())
}

/// The height of each mass level's own pressure, `[levels * cells]`, read
/// between the two interfaces of its layer in ln p: where a field carried at
/// the mass level (temperature, humidity) actually is.  NaN for a column
/// with no usable interfaces.
pub fn mass_level_heights_into(
    interfaces: &Interfaces<'_>,
    p_mass: &[f64],
    stencil: &InterfaceStencil,
    cells: usize,
    out: &mut [f64],
) -> Result<(), String> {
    let nz = check_columns(interfaces, p_mass, stencil, cells)?;
    if out.len() != nz * cells {
        return Err(format!(
            "mass-level heights write {nz} x {cells} values; was given room for {}",
            out.len()
        ));
    }
    let mut ln_p = vec![0.0f64; nz + 1];
    for cell in 0..cells {
        let usable = stencil.interface_ln_pressures(|k| p_mass[k * cells + cell], &mut ln_p);
        for k in 0..nz {
            out[k * cells + cell] = if usable {
                let (lower, upper) = (ln_p[k], ln_p[k + 1]);
                let fraction = (ln(p_mass[k * cells + cell]) - lower) / (upper - lower);
                let z_lower = interfaces.metres(k * cells + cell);
                let z_upper = interfaces.metres((k + 1) * cells + cell);
                z_lower + (z_upper - z_lower) * fraction
            } else {
                f64::NAN
            };
        }
    }
    Ok(())
}

/// The height of ONE isobaric surface, `level_pa`, in every column.
///
/// `z_interface` is `[(levels + 1) * cells]` metres, `p_mass` is
/// `[levels * cells]` pascals, both bottom-up with the level index
/// outermost; `eta_mass` and `eta_interface` are the model's vertical
/// coordinate at the mass levels and at the interfaces.
pub fn isobaric_height_from_interfaces(
    z_interface: &[f64],
    p_mass: &[f64],
    eta_mass: &[f64],
    eta_interface: &[f64],
    cells: usize,
    level_pa: f64,
) -> Result<Vec<f32>, String> {
    let stencil = InterfaceStencil::new(eta_mass, eta_interface)?;
    let mut planes =
        isobaric_heights_from_interfaces(z_interface, 1.0, p_mass, &stencil, cells, &[level_pa])?;
    Ok(planes.pop().expect("one plane per level"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An isothermal hydrostatic column, where the height of any pressure is
    /// known exactly: `z = H ln(p_surface / p)`.
    fn isothermal_column(
        surface_pa: f64,
        top_pa: f64,
        levels: usize,
        scale_height_m: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let eta_interface: Vec<f64> = (0..=levels)
            .map(|k| 1.0 - (k as f64 / levels as f64).powf(1.3))
            .collect();
        let p_interface: Vec<f64> = eta_interface
            .iter()
            .map(|eta| top_pa + eta * (surface_pa - top_pa))
            .collect();
        let z_interface: Vec<f64> = p_interface
            .iter()
            .map(|p| scale_height_m * (surface_pa / p).ln())
            .collect();
        let eta_mass: Vec<f64> = eta_interface.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
        // The model's mass-level pressure: the mean of its two interfaces.
        let p_mass: Vec<f64> = p_interface.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
        (z_interface, p_mass, eta_mass, eta_interface)
    }

    #[test]
    fn an_isobaric_height_is_read_between_interfaces_not_off_layer_means() {
        let scale_height = 7_400.0;
        let (z_interface, p_mass, eta_mass, eta_interface) =
            isothermal_column(98_000.0, 1_500.0, 30, scale_height);
        let truth = scale_height * (98_000.0f64 / 50_000.0).ln();
        let found = isobaric_height_from_interfaces(
            &z_interface,
            &p_mass,
            &eta_mass,
            &eta_interface,
            1,
            50_000.0,
        )
        .expect("height");
        assert!(
            (f64::from(found[0]) - truth).abs() < 0.05,
            "{} against {truth}",
            found[0]
        );

        // The layer-mean pairing this replaces, for the size of the thing:
        // mean interface height against mass-level pressure, in ln p.
        let z_mean: Vec<f64> = z_interface.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
        let target = 50_000.0f64.ln();
        let layer = (0..p_mass.len() - 1)
            .find(|k| p_mass[*k].ln() >= target && p_mass[k + 1].ln() < target)
            .expect("bracket");
        let fraction = (target - p_mass[layer].ln()) / (p_mass[layer + 1].ln() - p_mass[layer].ln());
        let paired = z_mean[layer] + (z_mean[layer + 1] - z_mean[layer]) * fraction;
        assert!(
            paired - truth > 2.0,
            "the layer-mean pairing reads {:.2} m high here",
            paired - truth
        );
    }

    #[test]
    fn a_level_below_the_ground_has_no_height_and_bad_shapes_are_refused() {
        let (z_interface, p_mass, eta_mass, eta_interface) =
            isothermal_column(48_000.0, 1_500.0, 50, 7_400.0);
        let found = isobaric_height_from_interfaces(
            &z_interface,
            &p_mass,
            &eta_mass,
            &eta_interface,
            1,
            50_000.0,
        )
        .expect("height");
        assert!(found[0].is_nan(), "500 hPa is under a 480 hPa surface");
        assert!(
            isobaric_height_from_interfaces(&z_interface, &p_mass, &eta_mass, &eta_mass, 1, 5.0e4)
                .is_err()
        );
        assert!(
            isobaric_height_from_interfaces(&z_interface[1..], &p_mass, &eta_mass, &eta_interface, 1, 5.0e4)
                .is_err()
        );
        assert!(
            isobaric_height_from_interfaces(&z_interface, &p_mass, &eta_mass, &eta_interface, 1, -1.0)
                .is_err()
        );
    }

    /// Many levels in one pass are the same numbers, bit for bit, as one
    /// level at a time; geopotential over gravity is the same as heights;
    /// and pressures in hPa read the same surface as pressures in Pa.
    #[test]
    fn many_levels_geopotential_and_hectopascals_agree_with_one_level_in_metres() {
        let mut z = Vec::new();
        let mut p = Vec::new();
        let columns = [(101_000.0, 7_200.0), (84_000.0, 7_600.0), (70_500.0, 7_000.0)];
        let built: Vec<_> = columns
            .iter()
            .map(|&(surface, h)| isothermal_column(surface, 2_000.0, 40, h))
            .collect();
        let levels = 40;
        for k in 0..=levels {
            for column in &built {
                z.push(column.0[k]);
            }
        }
        for k in 0..levels {
            for column in &built {
                p.push(column.1[k]);
            }
        }
        // One vertical coordinate for every column, as on a model grid.
        let (eta_mass, eta_interface) = (&built[0].2, &built[0].3);
        let cells = columns.len();
        let stencil = InterfaceStencil::new(eta_mass, eta_interface).unwrap();
        assert_eq!(stencil, InterfaceStencil::from_interfaces(eta_interface).unwrap());
        let targets = [100_000.0, 85_000.0, 70_000.0, 50_000.0, 25_000.0, 10_000.0];
        let planes = isobaric_heights_from_interfaces(&z, 1.0, &p, &stencil, cells, &targets).unwrap();
        for (plane, &target) in planes.iter().zip(&targets) {
            let one =
                isobaric_height_from_interfaces(&z, &p, eta_mass, eta_interface, cells, target)
                    .unwrap();
            assert_eq!(
                plane.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                one.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        // 1000 and 850 hPa are under the two raised columns; 700 hPa is
        // above all three.
        for plane in &planes[..2] {
            assert!(plane[0].is_finite() && plane[1].is_nan() && plane[2].is_nan());
        }
        assert!(planes[2].iter().all(|value| value.is_finite()));
        for (cell, &(surface, h)) in columns.iter().enumerate() {
            let truth = h * (surface / 50_000.0f64).ln();
            assert!((f64::from(planes[3][cell]) - truth).abs() < 0.05);
        }

        let phi: Vec<f64> = z.iter().map(|value| value * STANDARD_GRAVITY).collect();
        let hpa: Vec<f64> = p.iter().map(|value| value / 100.0).collect();
        let targets_hpa: Vec<f64> = targets.iter().map(|value| value / 100.0).collect();
        let from_phi =
            isobaric_heights_from_interfaces(&phi, STANDARD_GRAVITY, &hpa, &stencil, cells, &targets_hpa)
                .unwrap();
        for (a, b) in from_phi.iter().flatten().zip(planes.iter().flatten()) {
            assert!(
                (a.is_nan() && b.is_nan()) || (a - b).abs() < 1.0e-3,
                "{a} against {b}"
            );
        }
    }

    #[test]
    fn a_bracket_scans_from_the_ground_and_a_dead_column_is_skipped() {
        let ln_p = [1000f64.ln(), 900f64.ln(), 800f64.ln()];
        let (below, fraction) = interface_bracket(&ln_p, 850f64.ln()).unwrap();
        assert_eq!(below, 1);
        let expected = (850f64.ln() - 900f64.ln()) / (800f64.ln() - 900f64.ln());
        assert!((fraction - expected).abs() < 1e-15);
        assert!(interface_bracket(&ln_p, 1010f64.ln()).is_none());
        assert!(interface_bracket(&ln_p, 800f64.ln()).is_none());

        let stencil = InterfaceStencil::from_interfaces(&[1.0, 0.5, 0.0]).unwrap();
        let mut ln_p = [0.0; 3];
        assert!(!stencil.interface_ln_pressures(|k| [f64::NAN, 500.0][k], &mut ln_p));
        let planes = isobaric_heights_from_interfaces(
            &[0.0, 1.0, 2.0],
            1.0,
            &[f64::NAN, 500.0],
            &stencil,
            1,
            &[700.0],
        )
        .unwrap();
        assert!(planes[0][0].is_nan());
        assert!(InterfaceStencil::from_interfaces(&[1.0, 0.0]).is_err());
        assert!(InterfaceStencil::new(&[0.75, 0.75], &[1.0, 0.5, 0.0]).is_err());
        assert!(isobaric_heights_from_interfaces(&[0.0, 1.0, 2.0], 0.0, &[600.0, 500.0], &stencil, 1, &[550.0]).is_err());
    }
}
