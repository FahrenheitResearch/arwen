//! Production Rust maps and a native per-hour scorecard.

use crate::station_overlay::{scalar_dots, StationDot};
use crate::verification::{
    mapped_fields, Artifact, VerificationReceipt, VerificationRequest, RADAR_QUANTITIES,
    STATION_QUANTITIES,
};
use crate::verification_io::{ArmData, GridData, RadarData};
use rustwx_core::{CanonicalField, FieldSelector, ModelId};
use rustwx_render::{Color, PngWriteOptions, RgbaImage};
use std::path::Path;

fn save(image: &RgbaImage, path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let temporary = path.with_file_name(format!(
        ".{}.partial.png",
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("verification")
    ));
    rustwx_render::save_rgba_png_profile_with_options(
        image,
        &temporary,
        &PngWriteOptions::default(),
    )
    .map_err(|e| e.to_string())?;
    #[cfg(windows)]
    if path.is_file() {
        std::fs::remove_file(path).map_err(|e| e.to_string())?;
    }
    std::fs::rename(&temporary, path).map_err(|e| e.to_string())?;
    Ok(())
}

pub fn scorecard(receipt: &VerificationReceipt, path: &Path) -> Result<(), String> {
    let labels = receipt
        .stations
        .first()
        .map(|s| s.arms.iter().map(|a| a.label.clone()).collect::<Vec<_>>())
        .or_else(|| {
            receipt
                .radar
                .first()
                .map(|s| s.arms.iter().map(|a| a.label.clone()).collect())
        })
        .unwrap_or_default();
    let width = 480 + labels.len() as u32 * 330;
    let height = 252 + (receipt.stations.len() + receipt.radar.len()) as u32 * 48;
    let background = Color::rgba(245, 247, 250, 255);
    let ink = Color::rgba(23, 35, 52, 255);
    let muted = Color::rgba(75, 89, 106, 255);
    let green = Color::rgba(17, 111, 74, 255);
    let mut image = RgbaImage::from_pixel(
        width,
        height,
        image::Rgba([background.r, background.g, background.b, 255]),
    );
    let text = |img: &mut RgbaImage, t: &str, x: i32, y: i32, c: Color| {
        rustwx_render::draw_text_line(img, t, x, y, c, 2)
    };
    text(&mut image, "Observation verification", 28, 24, ink);
    text(
        &mut image,
        &format!(
            "Valid {} UTC | {}",
            receipt.valid_time.trim_end_matches('Z'),
            receipt.domain
        ),
        28,
        58,
        muted,
    );
    text(
        &mut image,
        "Stations: raw T2; bias / RMSE / count",
        28,
        106,
        ink,
    );
    for (i, label) in labels.iter().enumerate() {
        text(&mut image, label, 450 + i as i32 * 330, 106, ink);
    }
    let mut y = 146;
    for row in &receipt.stations {
        let title = STATION_QUANTITIES
            .iter()
            .find(|(q, _, _)| *q == row.quantity)
            .map(|(_, t, _)| *t)
            .unwrap_or(&row.quantity);
        text(&mut image, &format!("{title} ({})", row.units), 28, y, ink);
        for (i, a) in row.arms.iter().enumerate() {
            let line = match (a.bias, a.rmse) {
                (Some(b), Some(r)) => format!(
                    "{b:+.3} / {r:.4} / {}{}",
                    a.count,
                    if row.winner.as_deref() == Some(&a.label) {
                        " *"
                    } else {
                        ""
                    }
                ),
                _ => format!("missing / missing / {}", a.count),
            };
            text(
                &mut image,
                &line,
                450 + i as i32 * 330,
                y,
                if row.winner.as_deref() == Some(&a.label) {
                    green
                } else {
                    ink
                },
            );
        }
        y += 48;
    }
    y += 12;
    text(
        &mut image,
        "Radar: FSS / paired neighbourhood count",
        28,
        y,
        ink,
    );
    y += 42;
    for row in &receipt.radar {
        let prefix = if row.quantity == "composite_reflectivity" {
            "Reflectivity"
        } else {
            "Rain"
        };
        text(
            &mut image,
            &format!(
                "{prefix} >= {:.0} {} | {:.1}x{:.1} km",
                row.threshold, row.units, row.actual_width_km, row.actual_height_km
            ),
            28,
            y,
            ink,
        );
        for (i, a) in row.arms.iter().enumerate() {
            let line = if row.status == "no-observed-events" {
                format!("no observed events / {}", a.count)
            } else {
                match a.fss {
                    Some(s) => format!(
                        "{s:.4} / {}{}",
                        a.count,
                        if row.winner.as_deref() == Some(&a.label) {
                            " *"
                        } else {
                            ""
                        }
                    ),
                    None => format!("missing / {}", a.count),
                }
            };
            text(
                &mut image,
                &line,
                450 + i as i32 * 330,
                y,
                if row.winner.as_deref() == Some(&a.label) {
                    green
                } else {
                    muted
                },
            );
        }
        y += 48;
    }
    text(
        &mut image,
        "* Unrounded scores choose winners; wind is speed; ties and no-event rows have no winner",
        28,
        height as i32 - 42,
        muted,
    );
    save(&image, path)
}

fn style(quantity: &str) -> Result<rustwx_products::viewer::StoreVariableStyle, String> {
    let (selector, units) = match quantity {
        "temperature_2m" => (
            FieldSelector::height_agl(CanonicalField::Temperature, 2),
            "K",
        ),
        "dewpoint_2m" => (FieldSelector::height_agl(CanonicalField::Dewpoint, 2), "K"),
        "wind_speed_10m" => (
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10),
            "m/s",
        ),
        "composite_reflectivity" => (
            FieldSelector::entire_atmosphere(CanonicalField::CompositeReflectivity),
            "dBZ",
        ),
        "precipitation_1h" => (
            FieldSelector::surface(CanonicalField::TotalPrecipitation),
            "mm",
        ),
        _ => return Err(format!("no production style metadata for {quantity}")),
    };
    rustwx_products::viewer::operational_style_for_store_variable(
        if quantity == "precipitation_1h" {
            "apcp_1h"
        } else {
            quantity
        },
        &serde_json::to_value(selector).map_err(|e| e.to_string())?,
        units,
        ModelId::WrfGdex,
    )
    .ok_or_else(|| format!("no production colour table for {quantity}"))
}

pub fn map_sheets(
    request: &VerificationRequest,
    receipt: &VerificationReceipt,
    arms: &[ArmData],
    radars: &[RadarData],
) -> Result<Vec<Artifact>, String> {
    let Some(root) = &request.out_root else {
        return Ok(Vec::new());
    };
    let Some(grid) = arms.first().and_then(|a| a.grid.as_ref()) else {
        return Ok(Vec::new());
    };
    let day = &request.valid_time[..10];
    let stamp = request.valid_time.replace(['-', ':', 'T', 'Z'], "");
    let domain = crate::panel::safe_component(&request.domain, "native_grid");
    let products = STATION_QUANTITIES
        .iter()
        .map(|(q, t, _)| (*q, *t))
        .chain(RADAR_QUANTITIES.iter().map(|(q, t, _, _)| (*q, *t)));
    let mut artifacts = Vec::new();
    for (quantity, title) in products {
        if !arms
            .iter()
            .all(|a| a.grid.is_some() && a.fields.contains_key(quantity))
        {
            continue;
        }
        let style = style(quantity)?;
        let folder = root
            .join(&domain)
            .join(format!("verify_{quantity}"))
            .join(day);
        std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
        let mut images = Vec::new();
        let mut panel_paths = Vec::new();
        let mut unit_rows = Vec::new();
        let mut rows: Vec<(String, Vec<f64>, Option<usize>, String)> = Vec::new();
        for (i, a) in arms.iter().enumerate() {
            rows.push((
                a.label.clone(),
                mapped_fields(grid, a.grid.as_ref().unwrap(), &a.fields[quantity])?,
                Some(i),
                request.valid_time.trim_end_matches('Z').to_string(),
            ));
        }
        if let Some(r) = radars.iter().find(|r| r.quantity == quantity) {
            let masked = r
                .values
                .iter()
                .zip(&r.valid)
                .map(|(&v, &ok)| if ok { v } else { f64::NAN })
                .collect::<Vec<_>>();
            let label = r.provenance["source"]
                .as_str()
                .unwrap_or("observed")
                .to_uppercase();
            let offset = (crate::verification::parse_time(&r.valid_time)?
                - crate::verification::parse_time(&request.valid_time)?)
            .num_seconds();
            rows.push((
                label,
                mapped_fields(grid, &r.grid, &masked)?,
                None,
                format!(
                    "{} UTC | offset {offset:+} s",
                    r.valid_time.trim_end_matches('Z')
                ),
            ));
        }
        for (i, (label, values, arm_index, valid_label)) in rows.into_iter().enumerate() {
            let raw_range = finite_range(values.iter().copied());
            let display_range = finite_range(
                values
                    .iter()
                    .map(|&value| f64::from(style.convert.apply(value as f32))),
            );
            unit_rows.push(serde_json::json!({"label":label,"valid_time":valid_label,"canonical_range":raw_range,"display_range":display_range}));
            let path = folder.join(format!(".panel-{stamp}-{i}.png"));
            let dots: Vec<_> = receipt
                .station_samples
                .iter()
                .filter_map(|s| {
                    let observed = *s.observation.values.get(quantity)?;
                    if !s.forecast.iter().all(|f| f.values.contains_key(quantity)) {
                        return None;
                    }
                    let forecast =
                        arm_index.and_then(|j| s.forecast[j].values.get(quantity).copied());
                    if forecast.is_none() {
                        return None;
                    }
                    Some(StationDot {
                        latitude: s.observation.latitude,
                        longitude: s.observation.longitude,
                        observed: f64::from(style.convert.apply(observed as f32)),
                        forecast: forecast.map(|v| f64::from(style.convert.apply(v as f32))),
                    })
                })
                .collect();
            let overlays =
                scalar_dots(&dots, &style.display_units, request.station_mode == "error")?;
            render_one(
                grid,
                values,
                &label,
                title,
                request,
                &style,
                Some(&overlays),
                &path,
                &valid_label,
            )?;
            let decoded = image::open(&path).map_err(|e| e.to_string())?.to_rgba8();
            images.push(decoded);
            panel_paths.push(path);
        }
        let header = crate::compare::SheetHeader {
            title: format!("{title} ({})", style.display_units),
            subtitle: if STATION_QUANTITIES.iter().any(|(q, _, _)| *q == quantity) {
                format!(
                    "Valid {} UTC | {} | {} station dots",
                    request.valid_time.trim_end_matches('Z'),
                    request.domain,
                    request.station_mode
                )
            } else {
                format!(
                    "Valid {} UTC | {}",
                    request.valid_time.trim_end_matches('Z'),
                    request.domain
                )
            },
        };
        let sheet = crate::compare::compose_sheet(&images, &header)?;
        let sheet = if request.station_mode == "error"
            && STATION_QUANTITIES.iter().any(|(q, _, _)| *q == quantity)
        {
            crate::station_overlay::append_error_key(
                &sheet,
                &style.display_units,
                style.colormap_options,
            )?
        } else {
            sheet
        };
        let output = folder.join(format!("verification_{stamp}.png"));
        save(&sheet, &output)?;
        for path in panel_paths {
            std::fs::remove_file(&path)
                .map_err(|e| format!("remove created panel {}: {e}", path.display()))?;
        }
        let image_artifact = Artifact::new(format!("verify_{quantity}"), output.clone())?;
        if quantity == "precipitation_1h" {
            let cmap = rustwx_render::build_colormap(&style.scale, style.colormap_options);
            let physical_reference = f64::from(style.convert.apply(25.4));
            let map_color = cmap.map(physical_reference);
            let legend_color = rustwx_render::legend_color_at_rel(
                &cmap,
                style.legend_mode,
                rustwx_render::legend_tick_rel(&cmap, physical_reference)
                    .ok_or("precipitation reference has no legend position")?,
            );
            let rgba = |color: rustwx_render::Rgba| [color.r, color.g, color.b, color.a];
            let metadata = serde_json::json!({"schema":"gpuwm.verify-visuals.units.v1","quantity":quantity,"canonical_units":"mm","display_units":style.display_units,"conversion":"mm / 25.4","palette_levels":cmap.legend_levels_for_display(),"reference":{"canonical_value":25.4,"display_value":physical_reference,"map_rgba":rgba(map_color),"legend_rgba":rgba(legend_color)},"panels":unit_rows,"image_sha256":image_artifact.sha256});
            let metadata_path = folder.join(format!("verification_{stamp}.units.json"));
            std::fs::write(
                &metadata_path,
                serde_json::to_vec_pretty(&metadata).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            artifacts.push(Artifact::new(
                format!("verify_{quantity}_units"),
                metadata_path,
            )?);
        }
        artifacts.push(image_artifact);
    }
    Ok(artifacts)
}

fn finite_range(values: impl Iterator<Item = f64>) -> Option<[f64; 2]> {
    let mut range: Option<[f64; 2]> = None;
    for value in values.filter(|value| value.is_finite()) {
        range = Some(match range {
            Some([lo, hi]) => [f64::min(lo, value), f64::max(hi, value)],
            None => [value, value],
        });
    }
    range
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hourly_precipitation_keeps_physical_inches_in_fill_and_legend() {
        let native = style("precipitation_1h").unwrap();
        assert_eq!(native.display_units, "in");
        assert_eq!(
            native.convert,
            rustwx_products::viewer::UnitConvert::MmToInches
        );
        assert_eq!(native.convert.apply(25.4), 1.0);
        let selector =
            serde_json::to_value(FieldSelector::surface(CanonicalField::TotalPrecipitation))
                .unwrap();
        let alias = rustwx_products::viewer::operational_style_for_store_variable(
            "apcp_1h",
            &selector,
            "kg/m^2",
            ModelId::WrfGdex,
        )
        .unwrap();
        assert_eq!(native.convert, alias.convert);
        assert_eq!(native.scale, alias.scale);
        let cmap = rustwx_render::build_colormap(&native.scale, native.colormap_options);
        let display = f64::from(native.convert.apply(25.4));
        let map = cmap.map(display);
        let legend = rustwx_render::legend_color_at_rel(
            &cmap,
            native.legend_mode,
            rustwx_render::legend_tick_rel(&cmap, display).unwrap(),
        );
        assert_eq!(map, legend);
        assert_eq!(map, cmap.map(1.0));
        assert_ne!(map, cmap.map(25.4));
        assert!(cmap.legend_levels_for_display().contains(&1.0));
    }
}

fn render_one(
    grid: &GridData,
    values: Vec<f64>,
    label: &str,
    title: &str,
    request: &VerificationRequest,
    style: &rustwx_products::viewer::StoreVariableStyle,
    overlays: Option<&crate::annotate::MapOverlays>,
    path: &Path,
    valid_label: &str,
) -> Result<(), String> {
    crate::panel::render_panel(crate::panel::PanelRequest {
        lat_deg: &grid.lat,
        lon_deg: &grid.lon,
        projection: grid.projection.as_ref(),
        ny: grid.ny,
        nx: grid.nx,
        values: values
            .into_iter()
            .map(|v| style.convert.apply(v as f32))
            .collect(),
        product_slug: format!("verify_{}", crate::panel::safe_component(title, "field")),
        title: label.into(),
        display_units: style.display_units.clone(),
        scale: style.scale.clone(),
        cbar_tick_step: style.cbar_tick_step,
        legend: style.colormap_options.legend,
        render_density: style.colormap_options.render_density,
        subtitle_left: valid_label.into(),
        subtitle_center: None,
        subtitle_right: request.domain.clone(),
        width: 1100,
        height: 850,
        contours: Vec::new(),
        colorbar: true,
        overlays,
        annotations: None,
        out_path: path.into(),
    })?;
    Ok(())
}
