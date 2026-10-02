//! Align an earlier accumulated field to a moving grid's current ground.
//! Newly exposed points have no earlier measurement and remain missing.

use static_fields::projection::{GridSpec, ProjectedGrid, ProjectionKind};

use crate::error::{refuse, Result};
use crate::frame::FileMeta;
use crate::grid;

enum Inverse {
    Projected(ProjectedGrid),
    Regular { lat0: f64, dlat: f64, lon0: f64, dlon: f64, nx: usize, periodic: bool },
}

fn wrap180(value: f64) -> f64 {
    let wrapped = (value + 180.0).rem_euclid(360.0) - 180.0;
    if wrapped == -180.0 { 180.0 } else { wrapped }
}

impl Inverse {
    fn index(&self, lat: f64, lon: f64) -> (f64, f64) {
        match self {
            Self::Projected(grid) => {
                let (i, j) = grid.latlon_to_ij(lat, lon);
                (i - 1.0, j - 1.0)
            }
            Self::Regular { lat0, dlat, lon0, dlon, nx, periodic } => {
                let centre = lon0 + (*nx as f64 - 1.0) * dlon / 2.0;
                let i = (centre + wrap180(lon - centre) - lon0) / dlon;
                (if *periodic { i.rem_euclid(*nx as f64) } else { i }, (lat - lat0) / dlat)
            }
        }
    }

    fn periodic(&self) -> bool { matches!(self, Self::Regular { periodic: true, .. }) }
}

/// Previous cumulative precipitation evaluated on the current frame's
/// geographic points. Bilinear interpolation uses the previous model grid;
/// integer moves snap to exact cells within the coordinate-storage error.
pub fn align(
    meta: &FileMeta,
    old_lat: &[f64],
    old_lon: &[f64],
    now_lat: &[f64],
    now_lon: &[f64],
    previous: &[f64],
) -> Result<Vec<f64>> {
    let (nx, ny) = (meta.nx, meta.ny);
    let cells = nx * ny;
    if [old_lat.len(), old_lon.len(), now_lat.len(), now_lon.len(), previous.len()].iter().any(|&n| n != cells) {
        return Err(refuse("a moving grid's accumulation state has a different shape from its coordinates, so rain intervals could not be aligned to the same ground"));
    }
    let native = grid::describe(meta, old_lat, old_lon);
    let inverse = if let Some(regular) = native.regular {
        Inverse::Regular {
            lat0: old_lat[0], dlat: (old_lat[(ny - 1) * nx] - old_lat[0]) / (ny - 1) as f64,
            lon0: old_lon[0], dlon: wrap180(old_lon[1] - old_lon[0]), nx, periodic: regular.periodic,
        }
    } else {
        let kind = match meta.map_proj {
            1 => ProjectionKind::Lambert,
            2 => ProjectionKind::Polar,
            3 => ProjectionKind::Mercator,
            other => return Err(refuse(format!(
                "moving MAP_PROJ {other} has no accumulation-grid inverse here, so subtracting earlier array indices would publish rain from different ground"
            ))),
        };
        Inverse::Projected(ProjectedGrid::new(GridSpec {
            kind, ref_lat: old_lat[0], ref_lon: old_lon[0], truelat1: meta.truelat1,
            truelat2: meta.truelat2, stand_lon: meta.stand_lon, dx: meta.dx, dy: meta.dy,
            e_we: nx as i64 + 1, e_sn: ny as i64 + 1, known_x: 1.0, known_y: 1.0,
            moad_cen_lat: meta.moad_cen_lat, moad_cen_lon: meta.cen_lon,
            lat_deg: vec![], lon0_deg: 0.0, dlon_deg: 0.0,
        }).map_err(|e| refuse(format!("the earlier moving grid cannot be reconstructed ({e}), so its accumulated rain cannot be placed on the current ground")))?)
    };
    let periodic = inverse.periodic();
    for c in 0..cells {
        let (i, j) = inverse.index(old_lat[c], old_lon[c]);
        let di = if periodic {
            let difference = (i - (c % nx) as f64).rem_euclid(nx as f64);
            difference.min(nx as f64 - difference)
        } else { (i - (c % nx) as f64).abs() };
        if !i.is_finite() || !j.is_finite() || di.max((j - (c / nx) as f64).abs()) > 0.01 {
            return Err(refuse("the earlier moving grid's inverse disagrees with its own coordinates by more than 0.01 cell, so rain intervals would subtract different ground"));
        }
    }
    let snap = |value: f64| if (value - value.round()).abs() <= 0.01 { value.round() } else { value };
    let mut output = Vec::with_capacity(cells);
    for c in 0..cells {
        let (i, j) = inverse.index(now_lat[c], now_lon[c]);
        let (i, j) = (snap(i), snap(j));
        if !i.is_finite() || !j.is_finite() || j < 0.0 || j > (ny - 1) as f64
            || (!periodic && (i < 0.0 || i > (nx - 1) as f64)) {
            output.push(f64::NAN);
            continue;
        }
        let floor_i = i.floor();
        let i0 = if periodic { (floor_i as i64).rem_euclid(nx as i64) as usize } else { floor_i as usize };
        let i1 = if periodic { (i0 + 1) % nx } else { (i0 + 1).min(nx - 1) };
        let j0 = j.floor() as usize;
        let j1 = (j0 + 1).min(ny - 1);
        let (fx, fy) = (i - floor_i, j - j0 as f64);
        let mut value = 0.0;
        for (index, weight) in [
            (j0 * nx + i0, (1.0 - fx) * (1.0 - fy)),
            (j0 * nx + i1, fx * (1.0 - fy)),
            (j1 * nx + i0, (1.0 - fx) * fy),
            (j1 * nx + i1, fx * fy),
        ] {
            if weight > 0.0 { value += weight * previous[index]; }
        }
        output.push(value);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(map_proj: i32) -> FileMeta {
        FileMeta {
            nx: 8, ny: 7, nz: 2, domain: "d01".into(), parent: None, map_proj,
            truelat1: 30.0, truelat2: 60.0, stand_lon: -98.0, cen_lat: 38.0, cen_lon: -98.0,
            moad_cen_lat: 38.0, pole_lat: 90.0, pole_lon: 0.0, dx: 12_000.0, dy: 12_000.0,
            simulation_start: None, gpuwm_version: None, ic_source: None, ic_cycle: None,
            source_engine: None,
            history_preset: None, spec_bdy_width: None, bucket_mm: None,
        }
    }

    fn coordinates(meta: &FileMeta, kind: ProjectionKind, shift: f64) -> (Vec<f64>, Vec<f64>) {
        let grid = ProjectedGrid::new(GridSpec {
            kind, ref_lat: 38.0, ref_lon: -98.0, truelat1: meta.truelat1, truelat2: meta.truelat2,
            stand_lon: meta.stand_lon, dx: meta.dx, dy: meta.dy,
            e_we: meta.nx as i64 + 1, e_sn: meta.ny as i64 + 1,
            known_x: (meta.nx as f64 + 1.0) / 2.0 - shift,
            known_y: (meta.ny as f64 + 1.0) / 2.0,
            moad_cen_lat: meta.moad_cen_lat, moad_cen_lon: meta.cen_lon,
            lat_deg: vec![], lon0_deg: 0.0, dlon_deg: 0.0,
        }).unwrap();
        let mut lat = Vec::new(); let mut lon = Vec::new();
        for j in 0..meta.ny { for i in 0..meta.nx {
            let (y, x) = grid.ij_to_latlon(i as f64 + 1.0, j as f64 + 1.0);
            lat.push(f64::from(y as f32)); lon.push(f64::from(x as f32));
        }}
        (lat, lon)
    }

    #[test]
    fn projected_east_and_west_moves_align_without_fabricating_rain() {
        for (map, kind) in [(1, ProjectionKind::Lambert), (2, ProjectionKind::Polar), (3, ProjectionKind::Mercator)] {
            let meta = meta(map);
            let (old_lat, old_lon) = coordinates(&meta, kind, 0.0);
            let previous: Vec<f64> = (0..meta.nx * meta.ny).map(|c| 10.0 * (c % meta.nx) as f64 + 100.0 * (c / meta.nx) as f64).collect();
            for shift in [-1.0, 1.0] {
                let (now_lat, now_lon) = coordinates(&meta, kind, shift);
                let values = align(&meta, &old_lat, &old_lon, &now_lat, &now_lon, &previous).unwrap();
                for c in 0..values.len() {
                    let old_i = (c % meta.nx) as i64 + shift as i64;
                    if (0..meta.nx as i64).contains(&old_i) {
                        assert_eq!(values[c], previous[(c / meta.nx) * meta.nx + old_i as usize], "projection {map}, shift {shift}");
                    } else { assert!(values[c].is_nan()); }
                }
            }
        }
    }

    #[test]
    fn regular_periodic_longitudes_wrap_and_missing_latitude_stays_missing() {
        let mut meta = meta(6); meta.nx = 4; meta.ny = 3;
        let mut old_lat = Vec::new(); let mut old_lon = Vec::new();
        let mut now_lat = Vec::new(); let mut now_lon = Vec::new();
        for j in 0..meta.ny { for i in 0..meta.nx {
            old_lat.push(-45.0 + 45.0 * j as f64); old_lon.push(-180.0 + 90.0 * i as f64);
            now_lat.push(45.0 * j as f64); now_lon.push(-90.0 + 90.0 * i as f64);
        }}
        let previous: Vec<f64> = (0..12).map(f64::from).collect();
        let values = align(&meta, &old_lat, &old_lon, &now_lat, &now_lon, &previous).unwrap();
        assert_eq!(&values[..4], &[5.0, 6.0, 7.0, 4.0]);
        assert_eq!(&values[4..8], &[9.0, 10.0, 11.0, 8.0]);
        assert!(values[8..].iter().all(|v| v.is_nan()));
    }
}
