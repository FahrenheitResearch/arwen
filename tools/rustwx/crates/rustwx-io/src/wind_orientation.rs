//! Physical native-grid wind orientation, shared with Desktop's qualified IO owner.
use super::{GridDefinition, GridMemo, ExtractedFieldValues, IoError};
use rustwx_core::{CanonicalField, FieldProduct, FieldSelector, ModelId, LatLonGrid, SelectedField2D, VerticalSelector};
use std::collections::{HashMap, HashSet, hash_map::Entry};

pub fn validate_regional_lambert_wind_grid(model: ModelId, grid: &GridDefinition) -> Result<(), IoError> {
    let finite = [grid.dx, grid.dy, grid.lat1, grid.lon1, grid.lad,
        grid.latin1, grid.latin2, grid.lov].iter().all(|value| value.is_finite());
    if grid.template != 30 || grid.shape_of_earth != 6 || grid.projection_center_flag != 0
        || grid.scan_mode != 0x40 || grid.resolution_flags & 0x08 == 0
        || grid.nx < 2 || grid.ny < 2 || grid.nx.checked_mul(grid.ny) != Some(grid.num_data_points)
        || !finite || grid.dx <= 0. || grid.dy <= 0. || grid.lat1.abs() >= 90.
        || !(0. ..90.).contains(&grid.latin1) || !(0. ..90.).contains(&grid.latin2)
        || grid.latin1 == 0. || grid.latin2 == 0. || (grid.lad - grid.latin1).abs() > 1e-6 {
        return Err(IoError::UnsafeGridRelativeWind { model, detail:
            "grid-relative winds require a finite northern spherical Lambert grid with positive i/j scanning, matching declared dimensions, positive spacing, and its first standard parallel as the grid-spacing latitude".into() });
    }
    Ok(())
}

pub(super) fn is_horizontal_wind_component(selector: FieldSelector) -> bool {
    matches!(
        selector.field,
        CanonicalField::UWind | CanonicalField::VWind
    )
}

fn paired_horizontal_wind_selector(selector: FieldSelector) -> Option<FieldSelector> {
    let field = match selector.field {
        CanonicalField::UWind => CanonicalField::VWind,
        CanonicalField::VWind => CanonicalField::UWind,
        _ => return None,
    };
    Some(FieldSelector {
        field,
        vertical: selector.vertical,
        product: selector.product,
    })
}

fn wind_pair_error(model: ModelId, selector: FieldSelector) -> IoError {
    let pair = paired_horizontal_wind_selector(selector)
        .expect("wind-pair errors are only constructed for U/V selectors");
    IoError::UnsafeGridRelativeWind {
        model,
        detail: format!(
            "selector '{selector}' cannot be normalized without its matching '{pair}' component"
        ),
    }
}

pub(super) fn rotate_regional_grid_relative_wind_values(
    model: ModelId,
    extracted: &mut [ExtractedFieldValues],
    grid_memo: &GridMemo,
    wind_rotation_grids: &HashSet<usize>,
) -> Result<(), IoError> {
    if wind_rotation_grids.is_empty() {
        return Ok(());
    }

    let mut pairs: HashMap<(VerticalSelector, FieldProduct), [Option<usize>; 2]> = HashMap::new();
    for (index, field) in extracted.iter().enumerate() {
        if !wind_rotation_grids.contains(&field.grid_index) { continue; }
        let slot = match field.selector.field {
            CanonicalField::UWind => 0,
            CanonicalField::VWind => 1,
            _ => continue,
        };
        let pair = pairs
            .entry((field.selector.vertical, field.selector.product))
            .or_insert([None, None]);
        if pair[slot].replace(index).is_some() {
            return Err(IoError::UnsafeGridRelativeWind {
                model,
                detail: format!("duplicate canonical selector '{}'", field.selector),
            });
        }
    }

    let mut coefficients: HashMap<usize, Vec<(f32, f32)>> = HashMap::new();
    for pair in pairs.values() {
        let (Some(u_index), Some(v_index)) = (pair[0], pair[1]) else {
            let index = pair[0].or(pair[1]).expect("pair contains one component");
            return Err(wind_pair_error(model, extracted[index].selector));
        };
        let u_grid_index = extracted[u_index].grid_index;
        let v_grid_index = extracted[v_index].grid_index;
        if u_grid_index != v_grid_index {
            return Err(IoError::UnsafeGridRelativeWind {
                model,
                detail: format!(
                    "wind pair '{}' and '{}' use different native grids",
                    extracted[u_index].selector, extracted[v_index].selector
                ),
            });
        }
        let grid = &grid_memo
            .slots
            .get(u_grid_index)
            .ok_or_else(|| IoError::UnsafeGridRelativeWind {
                model,
                detail: format!("wind pair references missing grid slot {u_grid_index}"),
            })?
            .0
            .grid;
        let coefficients = match coefficients.entry(u_grid_index) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                entry.insert(grid_i_to_earth_rotation_coefficients(model, grid)?)
            }
        };
        let (u, v) = two_mut(extracted, u_index, v_index);
        rotate_grid_relative_wind_pair(
            model,
            u.selector,
            &mut u.values,
            &mut v.values,
            coefficients,
        )?;
    }
    Ok(())
}

pub fn rotate_normalized_grid_relative_wind_fields_to_earth(
    model: ModelId,
    extracted: &mut [SelectedField2D],
) -> Result<(), IoError> {
    let mut pairs: HashMap<(VerticalSelector, FieldProduct), [Option<usize>; 2]> = HashMap::new();
    for (index, field) in extracted.iter().enumerate() {
        let slot = match field.selector.field {
            CanonicalField::UWind => 0,
            CanonicalField::VWind => 1,
            _ => continue,
        };
        let pair = pairs
            .entry((field.selector.vertical, field.selector.product))
            .or_insert([None, None]);
        if pair[slot].replace(index).is_some() {
            return Err(IoError::UnsafeGridRelativeWind {
                model,
                detail: format!("duplicate canonical selector '{}'", field.selector),
            });
        }
    }

    for pair in pairs.values() {
        let (Some(u_index), Some(v_index)) = (pair[0], pair[1]) else {
            let index = pair[0].or(pair[1]).expect("pair contains one component");
            return Err(wind_pair_error(model, extracted[index].selector));
        };
        let (u, v) = two_mut(extracted, u_index, v_index);
        if u.grid != v.grid || u.projection != v.projection {
            return Err(IoError::UnsafeGridRelativeWind {
                model,
                detail: format!(
                    "wind pair '{}' and '{}' use different native grids",
                    u.selector, v.selector
                ),
            });
        }
        let coefficients = grid_i_to_earth_rotation_coefficients(model, &u.grid)?;
        rotate_grid_relative_wind_pair(
            model,
            u.selector,
            &mut u.values,
            &mut v.values,
            &coefficients,
        )?;
    }
    Ok(())
}

fn two_mut<T>(values: &mut [T], first: usize, second: usize) -> (&mut T, &mut T) {
    debug_assert_ne!(first, second);
    if first < second {
        let (left, right) = values.split_at_mut(second);
        (&mut left[first], &mut right[0])
    } else {
        let (left, right) = values.split_at_mut(first);
        (&mut right[0], &mut left[second])
    }
}

fn grid_i_to_earth_rotation_coefficients(
    model: ModelId,
    grid: &LatLonGrid,
) -> Result<Vec<(f32, f32)>, IoError> {
    let nx = grid.shape.nx;
    let ny = grid.shape.ny;
    if nx < 2 || grid.lat_deg.len() != nx * ny || grid.lon_deg.len() != nx * ny {
        return Err(IoError::UnsafeGridRelativeWind {
            model,
            detail: format!(
                "normalized grid {}x{} cannot define a positive-i tangent",
                nx, ny
            ),
        });
    }

    let mut out = Vec::with_capacity(grid.shape.len());
    for row in 0..ny {
        for column in 0..nx {
            let center = row * nx + column;
            let before = row * nx + column.saturating_sub(1);
            let after = row * nx + (column + 1).min(nx - 1);
            let lat = f64::from(grid.lat_deg[center]).to_radians();
            let lon = f64::from(grid.lon_deg[center]).to_radians();
            let before_xyz = geographic_unit_vector(
                f64::from(grid.lat_deg[before]),
                f64::from(grid.lon_deg[before]),
            );
            let center_xyz = geographic_unit_vector(
                f64::from(grid.lat_deg[center]),
                f64::from(grid.lon_deg[center]),
            );
            let after_xyz = geographic_unit_vector(
                f64::from(grid.lat_deg[after]),
                f64::from(grid.lon_deg[after]),
            );
            let backward_delta = subtract3(center_xyz, before_xyz);
            let forward_delta = subtract3(after_xyz, center_xyz);
            let backward_length = norm3(backward_delta);
            let forward_length = norm3(forward_delta);
            // Per-row longitude normalization can move one part of a
            // non-cyclic regional row across the dateline. Values and
            // coordinates remain aligned, but that creates one artificial
            // adjacency between the original row endpoints. At either side
            // of that seam, use the short physical neighbor rather than
            // differentiating across a continent. Normal cells retain the
            // lower-noise centered tangent.
            let delta = if backward_length <= 1.0e-12 {
                forward_delta
            } else if forward_length <= 1.0e-12 {
                backward_delta
            } else if backward_length > forward_length * 4.0 {
                forward_delta
            } else if forward_length > backward_length * 4.0 {
                backward_delta
            } else {
                add3(backward_delta, forward_delta)
            };
            let east = [-lon.sin(), lon.cos(), 0.0];
            let north = [-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos()];
            let grid_i_east = dot3(delta, east);
            let grid_i_north = dot3(delta, north);
            let norm = grid_i_east.hypot(grid_i_north);
            if !norm.is_finite() || norm <= 1.0e-12 {
                return Err(IoError::UnsafeGridRelativeWind {
                    model,
                    detail: format!(
                        "normalized grid has no finite positive-i tangent at row {row}, column {column}"
                    ),
                });
            }
            out.push(((grid_i_east / norm) as f32, (grid_i_north / norm) as f32));
        }
    }
    Ok(out)
}

fn geographic_unit_vector(lat_deg: f64, lon_deg: f64) -> [f64; 3] {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
}

fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn add3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn subtract3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn norm3(vector: [f64; 3]) -> f64 {
    dot3(vector, vector).sqrt()
}

fn rotate_grid_relative_wind_pair(
    model: ModelId,
    selector: FieldSelector,
    grid_u: &mut [f32],
    grid_v: &mut [f32],
    coefficients: &[(f32, f32)],
) -> Result<(), IoError> {
    if grid_u.len() != grid_v.len() || grid_u.len() != coefficients.len() {
        return Err(IoError::UnsafeGridRelativeWind {
            model,
            detail: format!(
                "wind pair for '{selector}' has inconsistent value/grid lengths ({}, {}, {})",
                grid_u.len(),
                grid_v.len(),
                coefficients.len()
            ),
        });
    }
    for ((u, v), &(cos_angle, sin_angle)) in
        grid_u.iter_mut().zip(grid_v.iter_mut()).zip(coefficients)
    {
        if !u.is_finite() || !v.is_finite() {
            *u = f32::NAN;
            *v = f32::NAN;
            continue;
        }
        let grid_u = *u;
        let grid_v = *v;
        *u = grid_u * cos_angle - grid_v * sin_angle;
        *v = grid_u * sin_angle + grid_v * cos_angle;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires retained operational RRFS three-field subset; no network"]
    fn retained_rrfs_values_lane_preserves_earth_wind_contract() {
        let path = std::env::var_os("ARWEN_RRFS_FIXTURE").expect("retained RRFS fixture");
        let bytes = std::fs::read(path).unwrap();
        let selectors = [FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
            FieldSelector::height_agl(CanonicalField::UWind, 10), FieldSelector::height_agl(CanonicalField::VWind, 10)];
        let grib = crate::Grib2File::from_bytes(&bytes).unwrap();
        let source_grid = &grib.messages.iter().find(|message| message.grid.resolution_flags & 0x08 != 0).unwrap().grid;
        validate_regional_lambert_wind_grid(ModelId::Rrfs, source_grid).unwrap();
        validate_regional_lambert_wind_grid(ModelId::Gfs, source_grid).unwrap();
        for field in ["earth", "scan", "spacing"] {
            let mut changed = source_grid.clone();
            match field { "earth" => changed.shape_of_earth = 5, "scan" => changed.scan_mode = 0,
                _ => changed.dx = 0. }
            assert!(validate_regional_lambert_wind_grid(ModelId::Rrfs, &changed).is_err());
        }
        let mut direct = crate::extract_fields_partial_from_model_bytes_at_forecast_hour(
            ModelId::Rrfs, &bytes, None, &selectors, Some(0)).unwrap().extracted;
        let original_u = direct.iter().find(|field| field.selector == selectors[1]).unwrap().values.clone();
        let original_v = direct.iter().find(|field| field.selector == selectors[2]).unwrap().values.clone();
        rotate_normalized_grid_relative_wind_fields_to_earth(ModelId::Rrfs, &mut direct).unwrap();
        let native = crate::extract_field_values_partial_from_model_bytes_at_forecast_hour(
            ModelId::Rrfs, &bytes, None, &selectors, Some(0)).unwrap();
        assert!(native.missing.is_empty()); assert_eq!(native.extracted.len(), 3);
        for field in &native.extracted {
            let expected = direct.iter().find(|other| other.selector == field.selector).unwrap();
            assert_eq!(field.values.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
                expected.values.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        }
        let east = &native.extracted.iter().find(|field| field.selector == selectors[1]).unwrap().values;
        let north = &native.extracted.iter().find(|field| field.selector == selectors[2]).unwrap().values;
        assert!(original_u.iter().zip(east).any(|(u,e)| u.is_finite() && (u-e).abs() > 0.1));
        for ((u,v),(e,n)) in original_u.iter().zip(&original_v).zip(east.iter().zip(north)) {
            if [u,v,e,n].iter().all(|value|value.is_finite()) { assert!((u.hypot(*v)-e.hypot(*n)).abs() < 1e-4); }
        }
        assert!(crate::extract_field_values_partial_from_model_bytes_at_forecast_hour(
            ModelId::Rrfs, &bytes, None, &[selectors[1]], Some(0)).is_err());
        println!("retained RRFS: 3 fields, native wind rotation bit-exact with direct map; magnitude preserved; changed physical grid and missing V rejected");
    }
}
