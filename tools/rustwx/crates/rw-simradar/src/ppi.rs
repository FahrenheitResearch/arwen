//! PPI sweep sampling into the production rw_wrfbatch map renderer.
use crate::manifest::{self, Image};
use bowecho_simradar::radar_core::{ElevationCut, MomentType, RadarVolume};
use rustwx_render::{LegendControls, LegendMode, RenderDensity};
use rw_wrfbatch::{
    panel::{PanelRequest, render_panel},
    scales,
};
use std::path::Path;

const EARTH_M: f64 = 6_371_000.0;

fn destination(lat: f64, lon: f64, azimuth: f64, ground_m: f64) -> (f32, f32) {
    let (lat, lon) = (lat.to_radians(), lon.to_radians());
    let a = ground_m / EARTH_M;
    let y = (lat.sin() * a.cos() + lat.cos() * a.sin() * azimuth.cos()).asin();
    let x = lon + (azimuth.sin() * a.sin() * lat.cos()).atan2(a.cos() - lat.sin() * y.sin());
    (
        y.to_degrees() as f32,
        ((x.to_degrees() + 180.0).rem_euclid(360.0) - 180.0) as f32,
    )
}

fn azimuth_rows(cut: &ElevationCut, moment: &MomentType) -> Vec<(f64, usize)> {
    let Some(grid) = cut.moments.get(moment) else {
        return Vec::new();
    };
    let mut rows: Vec<_> = grid
        .radial_indices
        .iter()
        .enumerate()
        .filter_map(|(row, &radial)| {
            cut.radials
                .get(radial)
                .map(|r| (f64::from(r.azimuth_deg).rem_euclid(360.0), row))
        })
        .collect();
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    rows
}

/// Nearest measured ray, including the north crossing.
fn nearest_row(rows: &[(f64, usize)], azimuth: f64) -> Option<usize> {
    if rows.is_empty() {
        return None;
    }
    let right = rows.partition_point(|r| r.0 < azimuth) % rows.len();
    let left = (right + rows.len() - 1) % rows.len();
    let distance = |v: f64| ((v - azimuth + 180.0).rem_euclid(360.0) - 180.0).abs();
    Some(if distance(rows[left].0) <= distance(rows[right].0) {
        rows[left].1
    } else {
        rows[right].1
    })
}

pub fn render(
    root: &Path,
    domain: &str,
    generation: &str,
    volume: &RadarVolume,
    tilt_count: usize,
    width: u32,
    height: u32,
) -> Result<Vec<Image>, String> {
    render_labeled(
        root,
        domain,
        generation,
        volume,
        tilt_count,
        width,
        height,
        "WOOF simulated",
        None,
    )
}

pub fn render_labeled(
    root: &Path,
    domain: &str,
    generation: &str,
    volume: &RadarVolume,
    tilt_count: usize,
    width: u32,
    height: u32,
    label: &str,
    range_limit_m: Option<f64>,
) -> Result<Vec<Image>, String> {
    let lat = f64::from(volume.site.latitude_deg.ok_or("radar lacks latitude")?);
    let lon = f64::from(volume.site.longitude_deg.ok_or("radar lacks longitude")?);
    let stem = volume.volume_time.format("%Y%m%dT%H%M%SZ").to_string();
    let mut images = Vec::new();
    let mut tilts: Vec<f32> = volume.cuts.iter().map(|c| c.elevation_deg).collect();
    tilts.sort_by(f32::total_cmp);
    tilts.dedup_by(|a, b| (*a - *b).abs() < 0.01);
    tilts.truncate(tilt_count);
    // Display resolution changes only the image, never the volume gate data.
    let n = width.max(height).clamp(256, 1000) as usize;
    for (tilt_index, &elevation) in tilts.iter().enumerate() {
        for (field, moment, units) in [
            ("reflectivity", MomentType::Reflectivity, "dBZ"),
            ("velocity", MomentType::Velocity, "m s-1"),
        ] {
            let Some((sweep_index, cut)) = volume.cuts.iter().enumerate().find(|(_, c)| {
                (c.elevation_deg - elevation).abs() < 0.01 && c.moments.contains_key(&moment)
            }) else {
                continue;
            };
            let grid = &cut.moments[&moment];
            let rows = azimuth_rows(cut, &moment);
            if rows.is_empty() || grid.gate_range.gate_count == 0 {
                continue;
            }
            let range = f64::from(grid.gate_range.first_gate_m)
                + f64::from(grid.gate_range.gate_spacing_m)
                    * (grid.gate_range.gate_count - 1) as f64;
            let range = range_limit_m.map(|limit| range.min(limit)).unwrap_or(range);
            let mut lats = Vec::with_capacity(n * n);
            let mut lons = Vec::with_capacity(n * n);
            let mut values = Vec::with_capacity(n * n);
            for j in 0..n {
                for i in 0..n {
                    let east = (2.0 * i as f64 / (n - 1) as f64 - 1.0) * range;
                    let north = (2.0 * j as f64 / (n - 1) as f64 - 1.0) * range;
                    let ground = east.hypot(north);
                    let az = east.atan2(north);
                    let (y, x) = destination(lat, lon, az, ground);
                    lats.push(y);
                    lons.push(x);
                    let earth = EARTH_M * (4.0 / 3.0);
                    let arc = ground / earth;
                    let slant = earth * arc.sin() / (f64::from(elevation).to_radians() + arc).cos();
                    let gate = ((slant - f64::from(grid.gate_range.first_gate_m))
                        / f64::from(grid.gate_range.gate_spacing_m))
                    .round();
                    let value = if gate >= 0.0 && gate < grid.gate_range.gate_count as f64 {
                        nearest_row(&rows, az.to_degrees().rem_euclid(360.0))
                            .and_then(|row| grid.scaled_value(row, gate as usize))
                            .unwrap_or(f32::NAN)
                    } else {
                        f32::NAN
                    };
                    values.push(value);
                }
            }
            let path = root
                .join("radar")
                .join(domain)
                .join(&volume.site.id)
                .join(generation)
                .join("ppi")
                .join(field)
                .join(format!("{stem}_tilt-{tilt_index:02}.png"));
            let scale = if field == "reflectivity" {
                scales::reflectivity_scale()
            } else {
                scales::radial_velocity_scale(60.0)
            };
            render_panel(PanelRequest {
                lat_deg: &lats,
                lon_deg: &lons,
                projection: None,
                ny: n,
                nx: n,
                values,
                product_slug: format!("simulated_radar_{field}"),
                title: format!(
                    "{label} {} ({units})",
                    if field == "velocity" {
                        "radial velocity"
                    } else {
                        field
                    }
                ),
                display_units: units.into(),
                scale,
                cbar_tick_step: None,
                legend: LegendControls {
                    mode: LegendMode::SmoothRamp,
                    ..Default::default()
                },
                render_density: RenderDensity::default(),
                subtitle_left: format!(
                    "{} | {:.2} deg | {}",
                    volume.site.id,
                    elevation,
                    volume.volume_time.format("%Y-%m-%d %H:%M:%S UTC")
                ),
                subtitle_center: None,
                subtitle_right: domain.into(),
                width,
                height,
                contours: Vec::new(),
                colorbar: true,
                overlays: None,
                annotations: None,
                out_path: path.clone(),
            })?;
            images.push(Image {
                artifact: manifest::artifact(root, &path, "png")?,
                field: field.into(),
                tilt_index,
                sweep_index,
                elevation_deg: elevation,
            });
        }
    }
    Ok(images)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ray_lookup_wraps_at_north() {
        let rows = [(1.0, 7), (90.0, 8), (359.0, 9)];
        assert_eq!(nearest_row(&rows, 359.9), Some(9));
        assert_eq!(nearest_row(&rows, 0.9), Some(7));
    }
}
