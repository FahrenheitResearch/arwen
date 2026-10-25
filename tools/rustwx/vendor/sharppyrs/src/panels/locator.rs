//! Location map panel: the sounding-point locator as a full inset panel
//! (world basemap + crosshair), replacing the "Psbl Haz. Type" box
//! in the default window layout. Same drawing as the hodograph's corner
//! inset, scaled up to its own cell with the inset-row title framing.

use egui::{Align2, Color32, Painter, Pos2, Rect, Stroke, StrokeKind, Vec2};

use crate::derived::DerivedParams;
use crate::skewt::SkewTStyle;
use crate::utils::qc;
use crate::{LocationFootprint, Profile};

const MAP_FILL: Color32 = Color32::from_rgb(0x05, 0x09, 0x0b);
const MAP_POINT_COLOR: Color32 = Color32::from_rgb(0xFF, 0xDA, 0x00);
const MAP_FOOTPRINT_COLOR: Color32 = Color32::from_rgb(0x00, 0xD8, 0xE8);
const PT: f64 = 4.0 / 3.0;

fn map_extent(
    interior: Rect,
    lat: f64,
    lon: f64,
    footprint: Option<LocationFootprint>,
) -> (f64, f64, f64, f64) {
    let cos_lat = lat.to_radians().cos().max(0.35);
    let aspect = (interior.width() / interior.height()) as f64;
    let mut half_lat = 4.0f64;
    if let Some(box_) = footprint {
        let required_lat = (lat - box_.south).abs().max((box_.north - lat).abs()) * 1.15;
        let required_lon = (lon - box_.west).abs().max((box_.east - lon).abs()) * 1.15;
        half_lat = half_lat
            .max(required_lat)
            .max(required_lon * cos_lat / aspect.max(0.01));
    }
    let half_lon = half_lat * aspect / cos_lat;
    // Over open ocean the default view can hold no outline at all; widen it
    // until a coast or border is in frame (see `basemap::widen_until_outline`).
    let ext = super::basemap::widen_until_outline(
        super::basemap::Extent {
            west: lon - half_lon,
            south: lat - half_lat,
            east: lon + half_lon,
            north: lat + half_lat,
        },
        lon,
        lat,
    );
    (ext.west, ext.south, ext.east, ext.north)
}

/// Draw the location panel into `rect`.
#[allow(unused_variables)]
pub fn draw(painter: &Painter, rect: Rect, prof: &Profile, dv: &DerivedParams, style: &SkewTStyle) {
    painter.rect_filled(rect, 0.0, style.bg_color);
    painter.rect_stroke(
        rect.shrink(0.5),
        0.0,
        Stroke::new(1.0, style.fg_color),
        StrokeKind::Inside,
    );

    // Title band like the other insets (vendored inset title style).
    let hgt = rect.height() as f64;
    let title_px = ((7.0 + hgt * 0.0045) * PT) as f32;
    let title_h = (title_px * 1.6).ceil();
    painter.text(
        egui::pos2(rect.center().x, rect.min.y + title_h * 0.5),
        Align2::CENTER_CENTER,
        "Location",
        style.bold_font(title_px),
        style.fg_color,
    );
    painter.line_segment(
        [
            egui::pos2(rect.min.x, rect.min.y + title_h),
            egui::pos2(rect.max.x, rect.min.y + title_h),
        ],
        Stroke::new(1.0, style.fg_color),
    );

    let lat = prof.inner.station.latitude;
    let lon = prof.inner.station.longitude;
    let body = Rect::from_min_max(egui::pos2(rect.min.x, rect.min.y + title_h), rect.max);
    if !lat.is_finite() || !lon.is_finite() || lat.abs() > 90.0 || lon.abs() > 180.0 {
        painter.text(
            body.center(),
            Align2::CENTER_CENTER,
            "no location",
            style.regular_font(title_px),
            Color32::from_gray(0x88),
        );
        return;
    }

    let interior = body.shrink(4.0);
    if interior.width() < 40.0 || interior.height() < 30.0 {
        return;
    }
    let map_bg = if style.bg_color == Color32::BLACK {
        MAP_FILL
    } else {
        style.bg_color
    };
    painter.rect_filled(interior, 0.0, map_bg);
    let mp = painter.with_clip_rect(interior);

    // Aspect-correct extent centered on the sounding point.
    let footprint = prof.location_footprint();
    let (west, south, east, north) = map_extent(interior, lat, lon, footprint);
    let map_point = |plon: f64, plat: f64| -> Pos2 {
        Pos2::new(
            interior.min.x + ((plon - west) / (east - west) * interior.width() as f64) as f32,
            interior.min.y + ((north - plat) / (north - south) * interior.height() as f64) as f32,
        )
    };

    super::basemap::draw(
        &mp,
        super::basemap::Extent {
            west,
            south,
            east,
            north,
        },
        map_point,
        Stroke::new(1.0, style.fg_color),
    );

    // Area-mean footprint first, sounding point second. The translucent fill
    // keeps state outlines legible; the sampled bounds and center marker use
    // distinct colors so an area average can never be mistaken for a point.
    if let Some(box_) = footprint {
        let bounds = Rect::from_two_pos(
            map_point(box_.west, box_.north),
            map_point(box_.east, box_.south),
        );
        mp.rect_filled(
            bounds,
            0.0,
            Color32::from_rgba_unmultiplied(
                MAP_FOOTPRINT_COLOR.r(),
                MAP_FOOTPRINT_COLOR.g(),
                MAP_FOOTPRINT_COLOR.b(),
                28,
            ),
        );
        mp.rect_stroke(
            bounds,
            0.0,
            Stroke::new(1.5, MAP_FOOTPRINT_COLOR),
            StrokeKind::Inside,
        );
    }

    // Crosshair at the sounding point.
    let c = map_point(lon, lat);
    let marker = Stroke::new(1.4, MAP_POINT_COLOR);
    mp.circle(c, 4.0, map_bg, marker);
    mp.line_segment([c - Vec2::new(7.0, 0.0), c + Vec2::new(7.0, 0.0)], marker);
    mp.line_segment([c - Vec2::new(0.0, 7.0), c + Vec2::new(0.0, 7.0)], marker);

    // Lat/lon readout, bottom-left inside the map.
    let _ = qc(0.0);
    let ns = if lat >= 0.0 { "N" } else { "S" };
    let ew = if lon >= 0.0 { "E" } else { "W" };
    mp.text(
        egui::pos2(interior.min.x + 4.0, interior.max.y - 3.0),
        Align2::LEFT_BOTTOM,
        format!("{:.2}\u{b0}{ns} {:.2}\u{b0}{ew}", lat.abs(), lon.abs()),
        style.regular_font((title_px * 0.9).max(9.0)),
        MAP_POINT_COLOR,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn location_extent_expands_to_keep_a_large_sampled_box_visible() {
        let interior = Rect::from_min_size(Pos2::ZERO, Vec2::new(320.0, 200.0));
        let footprint = LocationFootprint::new(28.0, -108.0, 42.0, -92.0).unwrap();
        let (west, south, east, north) = map_extent(interior, 35.0, -100.0, Some(footprint));

        assert!(west < footprint.west);
        assert!(east > footprint.east);
        assert!(south < footprint.south);
        assert!(north > footprint.north);
    }

    #[test]
    fn point_location_keeps_the_existing_four_degree_latitude_extent() {
        let interior = Rect::from_min_size(Pos2::ZERO, Vec2::new(320.0, 200.0));
        let (_, south, _, north) = map_extent(interior, 35.0, -100.0, None);
        assert_eq!((south, north), (31.0, 39.0));
    }

    #[test]
    fn a_point_outside_conus_gets_a_basemap_in_its_location_panel() {
        // Production check 2026-10-24: 12N 102.3W (East Pacific) drew an
        // empty inset. Paris keeps the 4-degree view and has outlines in
        // it; the open-ocean point widens until the Mexican coast is in.
        let interior = Rect::from_min_size(Pos2::ZERO, Vec2::new(320.0, 200.0));
        let (w, s, e, n) = map_extent(interior, 48.8, 2.3, None);
        assert_eq!((s, n), (44.8, 52.8));
        assert!(super::super::basemap::has_outline_in(super::super::basemap::Extent {
            west: w,
            south: s,
            east: e,
            north: n,
        }));
        let (w, s, e, n) = map_extent(interior, 12.0, -102.3, None);
        assert!(n - s > 8.0, "open ocean widens: {s}..{n}");
        assert!(super::super::basemap::has_outline_in(super::super::basemap::Extent {
            west: w,
            south: s,
            east: e,
            north: n,
        }));
    }
}
