//! Upward-positive geometric W on fixed AGL surfaces. This reads only W
//! and geometry, so a named LES map never needs parcel or pressure volumes.

use rustwx_core::{FieldSelector, GridProjection, LatLonGrid};
use wrf_core::WrfFile;

use crate::wrf_process::{
    WrfHourFields, WrfProcessOptions, WrfProductGroup, core_field_selector, push_canonical_values,
};

const PLANES: &[(&str, f64)] = &[
    ("vertical_velocity_1km_agl", 1000.0),
    ("vertical_velocity_3km_agl", 3000.0),
];

fn valid(value: f64) -> bool {
    value.is_finite() && value.abs() < 1.0e30 && value > -9998.0
}

/// Linear interpolation between adjacent mass levels. Exact hits retain
/// their value; missing brackets and targets outside the column stay NaN.
fn interpolate(
    w: &[f64],
    z: &[f64],
    nz: usize,
    cells: usize,
    target: f64,
) -> Result<Vec<f32>, String> {
    let expected = nz.checked_mul(cells).ok_or("W volume shape overflow")?;
    if w.len() != expected || z.len() != expected || !target.is_finite() {
        return Err(format!(
            "W/AGL volumes require {expected} values: W {}, height {}, target {target}",
            w.len(),
            z.len()
        ));
    }
    let mut out = vec![f32::NAN; cells];
    for (cell, value) in out.iter_mut().enumerate() {
        for k in 0..nz {
            let i = k * cells + cell;
            if z[i] == target && valid(w[i]) {
                *value = w[i] as f32;
                break;
            }
        }
        if value.is_finite() {
            continue;
        }
        for k in 0..nz.saturating_sub(1) {
            let lo = k * cells + cell;
            let hi = lo + cells;
            if valid(z[lo])
                && valid(z[hi])
                && valid(w[lo])
                && valid(w[hi])
                && z[lo] < target
                && target < z[hi]
            {
                let f = (target - z[lo]) / (z[hi] - z[lo]);
                *value = (w[lo] + f * (w[hi] - w[lo])) as f32;
                break;
            }
        }
    }
    Ok(out)
}

pub(crate) fn push(
    fields: &mut WrfHourFields,
    file: &WrfFile,
    timeidx: usize,
    grid: &LatLonGrid,
    projection: Option<GridProjection>,
    options: &WrfProcessOptions,
) {
    let wanted: Vec<_> = PLANES
        .iter()
        .filter(|(name, _)| options.should_process("wa", Some(name), WrfProductGroup::Core))
        .collect();
    if wanted.is_empty() {
        return;
    }
    // The same destaggering and geometric AGL height used by wrf-core's
    // `wa` and `height_agl` diagnostics. No omega conversion or sign flip.
    let volumes = file
        .w_destag(timeidx)
        .map_err(|e| format!("W: {e}"))
        .and_then(|w| {
            file.height_agl(timeidx)
                .map(|z| (w, z))
                .map_err(|e| format!("AGL height: {e}"))
        });
    for (name, target) in wanted {
        let result = match &volumes {
            Ok((w, z)) => interpolate(w, z, file.nz, grid.shape.len(), *target),
            Err(err) => Err(err.clone()),
        };
        match result {
            Ok(values) => {
                let selector: FieldSelector = core_field_selector(name);
                push_canonical_values(
                    fields,
                    grid,
                    projection.clone(),
                    name,
                    selector,
                    "m s-1",
                    values,
                );
            }
            Err(err) => fields.notes.push(format!("{name} unavailable: {err}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_interpolation_exact_hits_missing_brackets_and_no_extrapolation() {
        let w = [-4., -2., 0., f64::NAN, -1., 2., 8., 10.];
        let z = [500., 1000., 1500., 500., 1500., 2000., 2500., 1500.];
        let v = interpolate(&w, &z, 2, 4, 1000.).unwrap();
        assert_eq!(v[0], -2.5);
        assert_eq!(v[1], -2.);
        assert!(v[2].is_nan());
        assert!(v[3].is_nan());
        assert!(interpolate(&[1., 1.0e36], &[500., 1500.], 2, 1, 1000.).unwrap()[0].is_nan());
        assert!(interpolate(&[1.], &[500., 1500.], 2, 1, 1000.).is_err());
    }
}
