//! Selected isobaric chart planes using the sounding path's exact interpolation.
//!
//! This reads the same native WRF variables and uses the existing bracket/lerp
//! implementation, but never builds or persists the 37-level sounding volumes.
//! Height is read between layer interfaces exactly as the sounding volume's
//! is ([`FrameHeights`]), with the same named mass-level fallback.
use rustwx_core::{CanonicalField as F, FieldSelector, VerticalSelector};
use wrf_core::{ComputeOpts, WrfFile, getvar};

use crate::wrf_volumes::{
    FrameHeights, IsoVolume, check_native_3d_output, heights_at_levels,
    interpolate_field_at_levels, preflight_iso_volume_shape, validate_earth_relative_uvmet,
};

fn levels(selectors: &[FieldSelector], field: F) -> Vec<u16> {
    let mut result: Vec<_> = selectors
        .iter()
        .filter_map(|selector| {
            if selector.field != field || !selector.product.is_default() {
                return None;
            }
            match selector.vertical {
                VerticalSelector::IsobaricHpa(hpa) if hpa > 0 => Some(hpa),
                _ => None,
            }
        })
        .collect();
    result.sort();
    result.dedup();
    result
}

/// The selected planes, and the note the import must carry when the heights
/// took the named mass-level fallback.
pub(crate) fn build_chart_planes(
    file: &WrfFile,
    timeidx: usize,
    cells: usize,
    selectors: &[FieldSelector],
    progress: &mut dyn FnMut(String),
) -> Result<(Vec<IsoVolume>, Option<String>), String> {
    if selectors.is_empty() {
        return Ok((Vec::new(), None));
    }
    let (nx, ny, nz) = (file.nx, file.ny, file.nz);
    if nx.checked_mul(ny) != Some(cells) {
        return Err("WRF chart grid dimensions disagree".into());
    }
    // Conservative existing native working-set admission also bounds this
    // smaller path. It precedes the first native 3-D read.
    preflight_iso_volume_shape(nz, cells)?;
    let read = |name| {
        getvar(file, name, Some(timeidx), &ComputeOpts::default())
            .map_err(|error| format!("read WRF {name} for chart planes: {error}"))
    };
    progress("Reading pressure for selected 2-D chart planes".into());
    let pressure = read("pressure")?;
    let expected = check_native_3d_output(&pressure, "pressure", nz, ny, nx)?;
    let mut result = Vec::new();
    let mut height_note = None;
    for (field, variable, name, units) in [
        (F::Temperature, "temp", "temperature_iso", "K"),
        (F::Dewpoint, "td", "dewpoint_iso", "K"),
        (F::GeopotentialHeight, "height", "height_iso", "m"),
        (F::RelativeHumidity, "rh", "rh_chart_levels", "%"),
        (F::AbsoluteVorticity, "avo", "avo_chart_levels", "s-1"),
    ] {
        let requested = levels(selectors, field);
        if requested.is_empty() {
            continue;
        }
        progress(format!(
            "Computing {variable} at selected chart levels {requested:?} hPa"
        ));
        let field_result = (|| {
            let planes = if field == F::GeopotentialHeight {
                // Between layer interfaces, as the sounding volume reads it.
                let (heights, note) = FrameHeights::read(file, timeidx, "chart planes")?;
                if let Some(note) = &note {
                    progress(note.clone());
                }
                height_note = note;
                heights_at_levels(&pressure.data, &heights.view(), nz, cells, &requested)?
            } else {
                let mut output = read(variable)?;
                check_native_3d_output(&output, variable, nz, ny, nx)?;
                if field == F::Dewpoint {
                    for value in &mut output.data {
                        *value += 273.15;
                    }
                }
                interpolate_field_at_levels(&pressure.data, &output.data, nz, cells, &requested)?
            };
            Ok::<_, String>(IsoVolume {
                name: name.into(),
                units: units.into(),
                levels: planes,
            })
        })();
        match field_result {
            Ok(volume) => result.push(volume),
            Err(error) => progress(error),
        }
    }
    let u_levels = levels(selectors, F::UWind);
    let v_levels = levels(selectors, F::VWind);
    if !u_levels.is_empty() || !v_levels.is_empty() {
        progress("Computing earth-relative winds for selected chart levels".into());
        let wind_result = (|| {
            let output = read("uvmet")?;
            validate_earth_relative_uvmet(output, nz, ny, nx, expected)
        })();
        match wind_result {
            Ok(wind) => {
                for (name, requested, data) in [
                    ("u_iso", u_levels, &wind[..expected]),
                    ("v_iso", v_levels, &wind[expected..]),
                ] {
                    if !requested.is_empty() {
                        result.push(IsoVolume {
                            name: name.into(),
                            units: "m/s".into(),
                            levels: interpolate_field_at_levels(
                                &pressure.data,
                                data,
                                nz,
                                cells,
                                &requested,
                            )?,
                        });
                    }
                }
            }
            Err(error) => progress(error),
        }
    }
    Ok((result, height_note))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wrf_volumes::ColumnHeights;
    use rw_isobaric::{InterfaceStencil, STANDARD_GRAVITY};

    #[test]
    fn chart_levels_are_selected_without_a_full_pressure_volume() {
        let selectors = [
            FieldSelector::isobaric(F::Temperature, 850),
            FieldSelector::isobaric(F::Temperature, 850),
            FieldSelector::isobaric(F::UWind, 300),
        ];
        assert_eq!(levels(&selectors, F::Temperature), [850]);
        assert_eq!(levels(&selectors, F::UWind), [300]);
        assert!(levels(&selectors, F::Dewpoint).is_empty());
    }
    /// On the named mass-level height fallback every field, height
    /// included, is the one bracket-and-lerp walk.
    #[test]
    fn selected_chart_interpolation_matches_the_existing_sounding_planes() {
        let pressure = [1000., 990., 700., 690., 400., 390., 100., 90.];
        let temperature = [295., 294., 275., 274., 250., 249., 215., 214.];
        let dewpoint = [285., 284., 265., 264., 235., 234., 190., 189.];
        let height = [100., 110., 3100., 3110., 7100., 7110., 16100., 16110.];
        let u = [1., 2., 11., 12., 21., 22., 31., 32.];
        let v = [3., 4., 13., 14., 23., 24., 33., 34.];
        let (full, _) = crate::wrf_volumes::try_interpolate_iso_volumes(
            &pressure,
            &temperature,
            &dewpoint,
            ColumnHeights::MassLevels(&height),
            &u,
            &v,
            4,
            2,
            &mut |_| {},
        )
        .unwrap();
        for (name, data) in [
            ("temperature_iso", &temperature),
            ("dewpoint_iso", &dewpoint),
            ("height_iso", &height),
            ("u_iso", &u),
            ("v_iso", &v),
        ] {
            let chart =
                interpolate_field_at_levels(&pressure, data, 4, 2, &[300, 500, 700, 850]).unwrap();
            let full = full.iter().find(|volume| volume.name == name).unwrap();
            for (level, values) in chart {
                let expected = &full.levels.iter().find(|(hpa, _)| *hpa == level).unwrap().1;
                assert_eq!(
                    values
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    /// Two isothermal columns on one eta grid, where the height of any
    /// pressure is `H ln(p_surface / p)`: interface geopotential, mass-level
    /// pressure at the mean of its two interfaces.
    fn isothermal_columns() -> (Vec<f64>, Vec<f64>, InterfaceStencil, [(f64, f64); 2]) {
        const LEVELS: usize = 30;
        let columns = [(1_000.0, 7_300.0), (850.0, 7_600.0)];
        let top = 20.0;
        let eta: Vec<f64> = (0..=LEVELS)
            .map(|k| 1.0 - (k as f64 / LEVELS as f64).powf(1.2))
            .collect();
        let interface_p =
            |k: usize, (surface, _): (f64, f64)| top + eta[k] * (surface - top);
        let mut geopotential = Vec::new();
        for k in 0..=LEVELS {
            for column in columns {
                let (surface, scale_height) = column;
                geopotential.push(
                    STANDARD_GRAVITY * scale_height * (surface / interface_p(k, column)).ln(),
                );
            }
        }
        let mut pressure = Vec::new();
        for k in 0..LEVELS {
            for column in columns {
                pressure.push(0.5 * (interface_p(k, column) + interface_p(k + 1, column)));
            }
        }
        let stencil = InterfaceStencil::from_interfaces(&eta).unwrap();
        (geopotential, pressure, stencil, columns)
    }

    /// The chart plane's height and the sounding volume's height are the
    /// same numbers read between the same interfaces, and both sit on the
    /// isothermal truth where the layer-mean pairing they replace reads
    /// metres high.
    #[test]
    fn chart_and_sounding_heights_are_read_between_the_same_interfaces() {
        let (geopotential, pressure, stencil, columns) = isothermal_columns();
        let nz = stencil.levels();
        let heights = ColumnHeights::Interfaces {
            interface: &geopotential,
            per_metre: STANDARD_GRAVITY,
            stencil: &stencil,
        };
        let flat = vec![1.0; nz * 2];
        let (full, _) = crate::wrf_volumes::try_interpolate_iso_volumes(
            &pressure,
            &flat,
            &flat,
            ColumnHeights::Interfaces {
                interface: &geopotential,
                per_metre: STANDARD_GRAVITY,
                stencil: &stencil,
            },
            &flat,
            &flat,
            nz,
            2,
            &mut |_| {},
        )
        .unwrap();
        let sounding = full.iter().find(|volume| volume.name == "height_iso").unwrap();
        let chart = heights_at_levels(&pressure, &heights, nz, 2, &[250, 500, 700, 850]).unwrap();
        for (level, values) in &chart {
            let expected = &sounding.levels.iter().find(|(hpa, _)| hpa == level).unwrap().1;
            assert_eq!(
                values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
            );
        }
        let at_500 = &chart.iter().find(|(hpa, _)| *hpa == 500).unwrap().1;
        let layer_means: Vec<f64> = (0..nz * 2)
            .map(|index| 0.5 * (geopotential[index] + geopotential[index + 2]) / STANDARD_GRAVITY)
            .collect();
        let paired = interpolate_field_at_levels(&pressure, &layer_means, nz, 2, &[500]).unwrap();
        for (cell, (surface, scale_height)) in columns.into_iter().enumerate() {
            let truth = scale_height * (surface / 500.0f64).ln();
            assert!(
                (f64::from(at_500[cell]) - truth).abs() < 0.05,
                "column {cell}: {} against {truth}",
                at_500[cell]
            );
            assert!(
                f64::from(paired[0].1[cell]) - truth > 1.0,
                "column {cell}: the layer-mean pairing reads {} m high",
                f64::from(paired[0].1[cell]) - truth
            );
        }
        // 250 hPa is aloft in both columns.
        let at_250 = &chart.iter().find(|(hpa, _)| *hpa == 250).unwrap().1;
        assert!(at_250.iter().all(|value| value.is_finite()));
    }
}
