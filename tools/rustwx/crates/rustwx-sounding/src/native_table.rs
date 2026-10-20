use std::io::Cursor;

use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use rustwx_render::{BasemapStyle, Color, load_styled_basemap_features_for};
use sharprs::Profile as SharprsProfile;
use sharprs::params::{cape, composites, indices};
use sharprs::render::ComputedParams;
use sharprs::winds;

use crate::bridge::{SoundingMetadata, VerifiedEcapeParcelParams, VerifiedEcapeParcels};
use crate::error::SoundingBridgeError;

const IMG_W: u32 = 2400;
const TITLE_H: i32 = 44;
const UPPER_H: i32 = 1120;
const TABLE_Y: i32 = TITLE_H + UPPER_H;
const TABLE_H: i32 = 636;
const LOCATOR_X: i32 = 1176;
const LOCATOR_Y: i32 = TITLE_H;
const LOCATOR_W: i32 = 504;
const LOCATOR_H: i32 = 560;
const TITLE_SCALE: u32 = 3;
const SECTION_SCALE: u32 = 4;
const LABEL_SCALE: u32 = 3;
const BODY_SCALE: u32 = 3;
const SAMPLE_BODY_SCALE: u32 = 3;
const UNIT_SCALE: u32 = 2;
const TEXT_SIZE_FACTOR: f32 = 1.42;

const BG: Color = Color {
    r: 7,
    g: 10,
    b: 16,
    a: 255,
};
const TITLE_BG: Color = Color {
    r: 18,
    g: 22,
    b: 31,
    a: 255,
};
const PANEL_BG: Color = Color {
    r: 10,
    g: 14,
    b: 22,
    a: 255,
};
const LINE: Color = Color {
    r: 58,
    g: 66,
    b: 82,
    a: 255,
};
const LINE_DIM: Color = Color {
    r: 34,
    g: 41,
    b: 54,
    a: 255,
};
const TEXT: Color = Color {
    r: 231,
    g: 235,
    b: 241,
    a: 255,
};
const MUTED: Color = Color {
    r: 145,
    g: 154,
    b: 168,
    a: 255,
};
const LABEL: Color = Color {
    r: 141,
    g: 214,
    b: 232,
    a: 255,
};
const GOOD: Color = Color {
    r: 96,
    g: 220,
    b: 132,
    a: 255,
};
const WATCH: Color = Color {
    r: 255,
    g: 210,
    b: 88,
    a: 255,
};
const ORANGE: Color = Color {
    r: 255,
    g: 151,
    b: 79,
    a: 255,
};
const DANGER: Color = Color {
    r: 255,
    g: 86,
    b: 95,
    a: 255,
};

pub(crate) fn replace_title_and_table(
    base_png: &[u8],
    profile: &SharprsProfile,
    params: &ComputedParams,
    ecape: &VerifiedEcapeParcels,
    metadata: &SoundingMetadata,
) -> Result<Vec<u8>, SoundingBridgeError> {
    let mut image = image::load_from_memory_with_format(base_png, ImageFormat::Png)?.to_rgba8();
    if image.width() < IMG_W || image.height() < (TABLE_Y + TABLE_H) as u32 {
        return encode_png(image).map_err(Into::into);
    }

    let data = build_table_data(profile, params, ecape);
    draw_title(&mut image, profile);
    draw_locator_map(&mut image, profile, metadata);
    draw_locator_summary(&mut image, &data, metadata);
    draw_table(&mut image, &data);
    encode_png(image).map_err(Into::into)
}

fn draw_centered_text_line(image: &mut RgbaImage, text: &str, y: i32, color: Color, scale: u32) {
    rustwx_render::draw_centered_text_line_with_factor(
        image,
        text,
        y,
        color,
        scale,
        TEXT_SIZE_FACTOR,
    );
}

fn draw_text_line(image: &mut RgbaImage, text: &str, x: i32, y: i32, color: Color, scale: u32) {
    rustwx_render::draw_text_line_with_factor(image, text, x, y, color, scale, TEXT_SIZE_FACTOR);
}

fn draw_right_text_line(
    image: &mut RgbaImage,
    text: &str,
    x_right: i32,
    y: i32,
    color: Color,
    scale: u32,
) {
    rustwx_render::draw_right_text_line_with_factor(
        image,
        text,
        x_right,
        y,
        color,
        scale,
        TEXT_SIZE_FACTOR,
    );
}

fn draw_title(image: &mut RgbaImage, profile: &SharprsProfile) {
    fill_rect(image, 0, 0, image.width() as i32, TITLE_H, TITLE_BG);
    hline(image, 0, image.width() as i32 - 1, TITLE_H - 1, LINE);

    let station = profile.station.station_id.trim();
    let valid = profile.station.datetime.trim();
    let title = match (station.is_empty(), valid.is_empty()) {
        (true, true) => "rustwx Sounding Analysis".to_string(),
        (false, true) => format!("rustwx Sounding Analysis - {station}"),
        (true, false) => format!("rustwx Sounding Analysis - {valid}"),
        (false, false) => format!("rustwx Sounding Analysis - {station} - {valid}"),
    };
    draw_centered_text_line(image, &title, 3, TEXT, TITLE_SCALE);
}

fn draw_locator_map(image: &mut RgbaImage, profile: &SharprsProfile, metadata: &SoundingMetadata) {
    let Some((lat, lon)) = sounding_location(profile, metadata) else {
        return;
    };

    let x = LOCATOR_X;
    let y = LOCATOR_Y;
    let w = LOCATOR_W;
    let h = LOCATOR_H;
    let map_x = x + 12;
    let map_y = y + 66;
    let map_w = w - 24;
    let map_h = h - 94;
    let bounds = locator_bounds(lat, lon, metadata, map_w, map_h);
    let is_box = is_box_sample(metadata);

    fill_rect(image, x, y, w, h, PANEL_BG);
    draw_rect_border(image, x, y, w, h, LINE, 1);
    draw_text_line(
        image,
        if is_box {
            "BOX FOOTPRINT"
        } else {
            "SOUNDING LOCATION"
        },
        x + 14,
        y + 14,
        LABEL,
        SECTION_SCALE,
    );
    let coord = format!("{lat:.3}, {lon:.3}");
    draw_right_text_line(image, &coord, x + w - 14, y + 20, MUTED, BODY_SCALE);

    fill_rect(image, map_x, map_y, map_w, map_h, BG);
    draw_rect_border(image, map_x, map_y, map_w, map_h, LINE_DIM, 1);
    draw_graticule(image, &bounds, map_x, map_y, map_w, map_h);
    draw_locator_basemap(image, &bounds, map_x, map_y, map_w, map_h);

    if is_box {
        draw_locator_box(
            image, metadata, &bounds, map_x, map_y, map_w, map_h, lat, lon,
        );
    }
    draw_locator_marker(image, &bounds, map_x, map_y, map_w, map_h, lat, lon);

    let caption = if is_box { "box mean" } else { "point sample" };
    draw_text_line(image, caption, x + 14, y + h - 36, MUTED, BODY_SCALE);
}

fn draw_locator_summary(image: &mut RgbaImage, data: &TableData, metadata: &SoundingMetadata) {
    let x = LOCATOR_X;
    let y = LOCATOR_Y + LOCATOR_H;
    let w = LOCATOR_W;
    let h = UPPER_H - LOCATOR_H;

    fill_rect(image, x, y, w, h, PANEL_BG);
    draw_rect_border(image, x, y, w, h, LINE, 1);

    draw_text_line(
        image,
        "SOUNDING SUMMARY",
        x + 14,
        y + 14,
        LABEL,
        SECTION_SCALE,
    );
    hline(image, x + 10, x + w - 10, y + 58, LINE_DIM);

    let sb = find_parcel_row(data, "Surface");
    let ml = find_parcel_row(data, "Mixed-Layer");
    let mu = find_parcel_row(data, "Most-Unstable");
    let sfc3 = find_shear_row(data, "Sfc-3km");
    let sfc6 = find_shear_row(data, "Sfc-6km");
    let eff = find_shear_row(data, "Eff Inflow");

    let sb_cape = sb.map(|row| row.cape).unwrap_or(f64::NAN);
    let sb_ecape = sb.map(|row| row.ecape).unwrap_or(f64::NAN);
    let ml_cape = ml.map(|row| row.cape).unwrap_or(f64::NAN);
    let mu_cape = mu.map(|row| row.cape).unwrap_or(f64::NAN);

    draw_summary_column(
        image,
        x + 14,
        y + 78,
        222,
        "ENERGY",
        &[
            (
                "SB CAPE".to_string(),
                fmt_unit(sb_cape, "J/kg", 0),
                cape_color(sb_cape),
            ),
            (
                "SB A-ECAPE".to_string(),
                fmt_unit(sb_ecape, "J/kg", 0),
                cape_color(sb_ecape),
            ),
            (
                "ML CAPE".to_string(),
                fmt_unit(ml_cape, "J/kg", 0),
                cape_color(ml_cape),
            ),
            (
                "MU CAPE".to_string(),
                fmt_unit(mu_cape, "J/kg", 0),
                cape_color(mu_cape),
            ),
            (
                "DCAPE".to_string(),
                fmt_unit(data.dcape, "J/kg", 0),
                cape_color(data.dcape),
            ),
        ],
    );

    let sb_lcl = sb.map(|row| row.lcl_m).unwrap_or(f64::NAN);
    let sb_lfc = sb.map(|row| row.lfc_m).unwrap_or(f64::NAN);
    let sb_el = sb.map(|row| row.el_m).unwrap_or(f64::NAN);
    draw_summary_column(
        image,
        x + 264,
        y + 78,
        222,
        "LEVELS",
        &[
            ("LCL".to_string(), fmt_unit(sb_lcl, "m", 0), TEXT),
            ("LFC".to_string(), fmt_unit(sb_lfc, "m", 0), TEXT),
            ("EL".to_string(), fmt_unit(sb_el, "m", 0), TEXT),
            (
                "WB Zero".to_string(),
                fmt_unit(data.wb_zero_m, "m", 0),
                GOOD,
            ),
            (
                "Freezing".to_string(),
                fmt_unit(data.freezing_level_m, "m", 0),
                TEXT,
            ),
        ],
    );

    let sfc3_srh = sfc3.map(|row| row.srh).unwrap_or(f64::NAN);
    let sfc3_shear = sfc3.map(|row| row.shear).unwrap_or(f64::NAN);
    let sfc6_shear = sfc6.map(|row| row.shear).unwrap_or(f64::NAN);
    let eff_srh = eff.map(|row| row.srh).unwrap_or(f64::NAN);
    draw_summary_column(
        image,
        x + 14,
        y + 282,
        472,
        "SHEAR / MOTION",
        &[
            (
                "0-3km SRH".to_string(),
                fmt_unit(sfc3_srh, "m2/s2", 0),
                srh_color(sfc3_srh),
            ),
            (
                "0-3km Shear".to_string(),
                fmt_unit(sfc3_shear, "kt", 0),
                shear_color(sfc3_shear),
            ),
            (
                "0-6km Shear".to_string(),
                fmt_unit(sfc6_shear, "kt", 0),
                shear_color(sfc6_shear),
            ),
            (
                "Eff SRH".to_string(),
                fmt_unit(eff_srh, "m2/s2", 0),
                srh_color(eff_srh),
            ),
        ],
    );

    draw_sample_info_panel(image, metadata, x + 14, y + 430, 472, 112);
}

fn draw_sample_info_panel(
    image: &mut RgbaImage,
    metadata: &SoundingMetadata,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
) {
    fill_rect(image, x, y, w, h, PANEL_BG);
    draw_rect_border(image, x, y, w, h, LINE, 1);
    draw_text_line(image, "SOURCE", x + 10, y + 10, LABEL, SECTION_SCALE);
    draw_right_text_line(
        image,
        "MODEL PROFILE",
        x + w - 10,
        y + 12,
        TEXT,
        LABEL_SCALE,
    );
    hline(image, x + 8, x + w - 8, y + 54, LINE_DIM);

    let mut yy = y + 60;
    let method = metadata
        .sample_method
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("point");
    draw_text_line(
        image,
        &format!("method  {}", method.replace('_', "-")),
        x + 10,
        yy,
        TEXT,
        SAMPLE_BODY_SCALE,
    );
    yy += 31;

    let lat_radius = finite_positive(metadata.box_radius_lat_deg);
    let lon_radius = finite_positive(metadata.box_radius_lon_deg);
    let box_text = match (lat_radius, lon_radius) {
        (Some(lat), Some(lon)) => format!("box     {:.2} x {:.2} deg", lat * 2.0, lon * 2.0),
        _ => "box     --".to_string(),
    };
    draw_text_line(image, &box_text, x + 10, yy, TEXT, SAMPLE_BODY_SCALE);
    yy += 31;

    let coord_text = match (metadata.latitude_deg, metadata.longitude_deg) {
        (Some(lat), Some(lon)) if lat.is_finite() && lon.is_finite() => {
            format!("center  {lat:.3}, {lon:.3}")
        }
        _ => "center  --".to_string(),
    };
    if yy <= y + h - 18 {
        draw_text_line(image, &coord_text, x + 10, yy, MUTED, SAMPLE_BODY_SCALE);
    }
}

fn draw_summary_column(
    image: &mut RgbaImage,
    x: i32,
    y: i32,
    w: i32,
    title: &str,
    rows: &[(String, String, Color)],
) {
    draw_text_line(image, title, x, y, LABEL, LABEL_SCALE);
    hline(image, x, x + w, y + 34, LINE_DIM);

    let mut yy = y + 48;
    for (label, value, color) in rows {
        draw_text_line(image, label, x, yy, LABEL, BODY_SCALE);
        draw_right_text_line(image, value, x + w, yy - 1, *color, BODY_SCALE);
        yy += 36;
    }
}

fn find_parcel_row<'a>(data: &'a TableData, label: &str) -> Option<&'a ParcelRow> {
    data.parcels
        .iter()
        .find(|row| row.label.eq_ignore_ascii_case(label))
}

fn find_shear_row<'a>(data: &'a TableData, label: &str) -> Option<&'a ShearRow> {
    data.shear_layers
        .iter()
        .find(|row| row.label.eq_ignore_ascii_case(label))
}

fn sounding_location(profile: &SharprsProfile, metadata: &SoundingMetadata) -> Option<(f64, f64)> {
    let lat = metadata
        .latitude_deg
        .filter(|value| value.is_finite())
        .unwrap_or(profile.station.latitude);
    let lon = metadata
        .longitude_deg
        .filter(|value| value.is_finite())
        .unwrap_or(profile.station.longitude);
    (lat.is_finite() && lon.is_finite()).then_some((lat, lon))
}

fn is_box_sample(metadata: &SoundingMetadata) -> bool {
    metadata
        .sample_method
        .as_deref()
        .map(|method| {
            method.eq_ignore_ascii_case("box_mean") || method.eq_ignore_ascii_case("box-mean")
        })
        .unwrap_or(false)
        || metadata
            .box_radius_lat_deg
            .is_some_and(|radius| radius.is_finite() && radius > 0.0)
        || metadata
            .box_radius_lon_deg
            .is_some_and(|radius| radius.is_finite() && radius > 0.0)
}

#[derive(Debug, Clone, Copy)]
struct MapBounds {
    west: f64,
    east: f64,
    south: f64,
    north: f64,
}

fn locator_bounds(
    lat: f64,
    lon: f64,
    metadata: &SoundingMetadata,
    map_w: i32,
    map_h: i32,
) -> MapBounds {
    let box_lat = finite_positive(metadata.box_radius_lat_deg).unwrap_or(0.0);
    let box_lon = finite_positive(metadata.box_radius_lon_deg).unwrap_or(0.0);
    let mut lat_half = if box_lat > 0.0 {
        (box_lat * 3.2).max(0.55)
    } else {
        1.25
    };
    let mut lon_half = if box_lon > 0.0 {
        (box_lon * 3.2).max(0.55)
    } else {
        1.25 / lat.to_radians().cos().abs().max(0.35)
    };

    let map_aspect = (map_w as f64 / map_h.max(1) as f64).max(0.25);
    let cos_lat = lat.to_radians().cos().abs().max(0.25);
    let projected_ratio = lon_half * cos_lat / lat_half.max(0.001);
    if projected_ratio < map_aspect {
        lon_half = lat_half * map_aspect / cos_lat;
    } else {
        lat_half = lon_half * cos_lat / map_aspect;
    }

    lat_half = lat_half.clamp(0.35, 8.0);
    lon_half = lon_half.clamp(0.35, 12.0);

    MapBounds {
        west: lon - lon_half,
        east: lon + lon_half,
        south: (lat - lat_half).max(-89.0),
        north: (lat + lat_half).min(89.0),
    }
}

fn finite_positive(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite() && *value > 0.0)
}

fn draw_graticule(image: &mut RgbaImage, bounds: &MapBounds, x: i32, y: i32, w: i32, h: i32) {
    let lat_span = (bounds.north - bounds.south).abs();
    let lon_span = (bounds.east - bounds.west).abs();
    let step = if lat_span.max(lon_span) <= 1.25 {
        0.25
    } else if lat_span.max(lon_span) <= 3.0 {
        0.5
    } else {
        1.0
    };

    let mut lat_line = (bounds.south / step).ceil() * step;
    while lat_line <= bounds.north {
        if let (Some((x0, y0)), Some((x1, y1))) = (
            project_lon_lat(bounds, x, y, w, h, bounds.west, lat_line),
            project_lon_lat(bounds, x, y, w, h, bounds.east, lat_line),
        ) {
            draw_line(image, x0, y0, x1, y1, LINE_DIM);
        }
        lat_line += step;
    }

    let mut lon_line = (bounds.west / step).ceil() * step;
    while lon_line <= bounds.east {
        if let (Some((x0, y0)), Some((x1, y1))) = (
            project_lon_lat(bounds, x, y, w, h, lon_line, bounds.south),
            project_lon_lat(bounds, x, y, w, h, lon_line, bounds.north),
        ) {
            draw_line(image, x0, y0, x1, y1, LINE_DIM);
        }
        lon_line += step;
    }
}

fn draw_locator_basemap(image: &mut RgbaImage, bounds: &MapBounds, x: i32, y: i32, w: i32, h: i32) {
    for layer in load_styled_basemap_features_for(BasemapStyle::White) {
        let color = if layer.width <= 1 {
            Color {
                r: 72,
                g: 82,
                b: 100,
                a: 255,
            }
        } else {
            Color {
                r: 128,
                g: 143,
                b: 164,
                a: 255,
            }
        };
        let width = layer.width.max(1).min(2) as i32;
        for line in layer.lines {
            for segment in line.windows(2) {
                let [(lon0, lat0), (lon1, lat1)] = [segment[0], segment[1]];
                let Some(((clon0, clat0), (clon1, clat1))) =
                    clip_lonlat_segment(lon0, lat0, lon1, lat1, bounds)
                else {
                    continue;
                };
                let (Some((px0, py0)), Some((px1, py1))) = (
                    project_lon_lat(bounds, x, y, w, h, clon0, clat0),
                    project_lon_lat(bounds, x, y, w, h, clon1, clat1),
                ) else {
                    continue;
                };
                draw_line_width(image, px0, py0, px1, py1, color, width);
            }
        }
    }
}

fn draw_locator_box(
    image: &mut RgbaImage,
    metadata: &SoundingMetadata,
    bounds: &MapBounds,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    lat: f64,
    lon: f64,
) {
    let lat_radius = finite_positive(metadata.box_radius_lat_deg).unwrap_or(0.0);
    let lon_radius = finite_positive(metadata.box_radius_lon_deg).unwrap_or(0.0);
    if lat_radius <= 0.0 || lon_radius <= 0.0 {
        return;
    }

    let west = lon - lon_radius;
    let east = lon + lon_radius;
    let south = lat - lat_radius;
    let north = lat + lat_radius;
    let corners = [
        (west, south, east, south),
        (east, south, east, north),
        (east, north, west, north),
        (west, north, west, south),
    ];
    for (lon0, lat0, lon1, lat1) in corners {
        if let Some(((clon0, clat0), (clon1, clat1))) =
            clip_lonlat_segment(lon0, lat0, lon1, lat1, bounds)
        {
            if let (Some((px0, py0)), Some((px1, py1))) = (
                project_lon_lat(bounds, x, y, w, h, clon0, clat0),
                project_lon_lat(bounds, x, y, w, h, clon1, clat1),
            ) {
                draw_line_width(image, px0, py0, px1, py1, ORANGE, 3);
            }
        }
    }
}

fn draw_locator_marker(
    image: &mut RgbaImage,
    bounds: &MapBounds,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    lat: f64,
    lon: f64,
) {
    let Some((px, py)) = project_lon_lat(bounds, x, y, w, h, lon, lat) else {
        return;
    };
    fill_circle(image, px, py, 9, DANGER);
    draw_circle(image, px, py, 12, TEXT);
    hline(image, px - 18, px - 5, py, TEXT);
    hline(image, px + 5, px + 18, py, TEXT);
    vline(image, px, py - 18, py - 5, TEXT);
    vline(image, px, py + 5, py + 18, TEXT);
    draw_text_line(
        image,
        "POI",
        (px + 16).min(x + w - 58),
        py - 18,
        TEXT,
        LABEL_SCALE,
    );
}

fn project_lon_lat(
    bounds: &MapBounds,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    lon: f64,
    lat: f64,
) -> Option<(i32, i32)> {
    if !lon.is_finite() || !lat.is_finite() {
        return None;
    }
    if lon < bounds.west || lon > bounds.east || lat < bounds.south || lat > bounds.north {
        return None;
    }
    let px = x as f64 + (lon - bounds.west) / (bounds.east - bounds.west).max(1.0e-9) * w as f64;
    let py = y as f64 + (bounds.north - lat) / (bounds.north - bounds.south).max(1.0e-9) * h as f64;
    Some((px.round() as i32, py.round() as i32))
}

fn clip_lonlat_segment(
    lon0: f64,
    lat0: f64,
    lon1: f64,
    lat1: f64,
    bounds: &MapBounds,
) -> Option<((f64, f64), (f64, f64))> {
    let (mut x0, mut y0, mut x1, mut y1) = (lon0, lat0, lon1, lat1);
    let mut code0 = outcode(x0, y0, bounds);
    let mut code1 = outcode(x1, y1, bounds);

    loop {
        if code0 | code1 == 0 {
            return Some(((x0, y0), (x1, y1)));
        }
        if code0 & code1 != 0 {
            return None;
        }

        let code_out = if code0 != 0 { code0 } else { code1 };
        let (x, y) = if code_out & 8 != 0 {
            if (y1 - y0).abs() < 1.0e-12 {
                return None;
            }
            (
                x0 + (x1 - x0) * (bounds.north - y0) / (y1 - y0),
                bounds.north,
            )
        } else if code_out & 4 != 0 {
            if (y1 - y0).abs() < 1.0e-12 {
                return None;
            }
            (
                x0 + (x1 - x0) * (bounds.south - y0) / (y1 - y0),
                bounds.south,
            )
        } else if code_out & 2 != 0 {
            if (x1 - x0).abs() < 1.0e-12 {
                return None;
            }
            (bounds.east, y0 + (y1 - y0) * (bounds.east - x0) / (x1 - x0))
        } else {
            if (x1 - x0).abs() < 1.0e-12 {
                return None;
            }
            (bounds.west, y0 + (y1 - y0) * (bounds.west - x0) / (x1 - x0))
        };

        if code_out == code0 {
            x0 = x;
            y0 = y;
            code0 = outcode(x0, y0, bounds);
        } else {
            x1 = x;
            y1 = y;
            code1 = outcode(x1, y1, bounds);
        }
    }
}

fn outcode(lon: f64, lat: f64, bounds: &MapBounds) -> u8 {
    let mut code = 0;
    if lon < bounds.west {
        code |= 1;
    } else if lon > bounds.east {
        code |= 2;
    }
    if lat < bounds.south {
        code |= 4;
    } else if lat > bounds.north {
        code |= 8;
    }
    code
}

fn draw_table(image: &mut RgbaImage, data: &TableData) {
    fill_rect(image, 0, TABLE_Y, image.width() as i32, TABLE_H, BG);
    hline(image, 0, image.width() as i32 - 1, TABLE_Y, LINE);

    let left = 20;
    let mid = 1120;
    let right = 1740;
    let bottom = TABLE_Y + TABLE_H - 1;

    fill_rect(
        image,
        left - 10,
        TABLE_Y + 10,
        mid - left - 10,
        TABLE_H - 20,
        PANEL_BG,
    );
    fill_rect(
        image,
        mid + 10,
        TABLE_Y + 10,
        right - mid - 20,
        TABLE_H - 20,
        PANEL_BG,
    );
    fill_rect(
        image,
        right + 10,
        TABLE_Y + 10,
        image.width() as i32 - right - 20,
        TABLE_H - 20,
        PANEL_BG,
    );
    vline(image, mid - 1, TABLE_Y + 14, bottom - 14, LINE_DIM);
    vline(image, right - 1, TABLE_Y + 14, bottom - 14, LINE_DIM);

    draw_parcels(image, left, TABLE_Y + 20, data);
    draw_storm_motions(image, left, TABLE_Y + 284, data);
    draw_lapse_rates(image, left + 430, TABLE_Y + 284, data);
    draw_shear(image, mid + 24, TABLE_Y + 20, data);
    draw_indices(image, right + 24, TABLE_Y + 20, data);
    draw_composites(image, right + 24, TABLE_Y + 378, data);
}

fn draw_parcels(image: &mut RgbaImage, x: i32, y: i32, data: &TableData) {
    section_title(image, "PARCELS", x, y, 1060);
    let header_y = y + 54;
    let row_y = header_y + 58;
    let cols = [
        x + 42,
        x + 162,
        x + 262,
        x + 354,
        x + 448,
        x + 542,
        x + 630,
        x + 724,
        x + 826,
        x + 928,
    ];

    draw_text_line(image, "PCL", x, header_y, LABEL, LABEL_SCALE);
    for (label, col) in [
        ("A-ECAPE", cols[1]),
        ("NCAPE", cols[2]),
        ("CAPE", cols[3]),
        ("3CAPE", cols[4]),
        ("6CAPE", cols[5]),
        ("CINH", cols[6]),
        ("LCL", cols[7]),
        ("LFC", cols[8]),
        ("EL", cols[9]),
    ] {
        draw_right_text_line(image, label, col, header_y, LABEL, LABEL_SCALE);
    }
    for (unit, col) in [
        ("J/kg", cols[1]),
        ("", cols[2]),
        ("J/kg", cols[3]),
        ("J/kg", cols[4]),
        ("J/kg", cols[5]),
        ("J/kg", cols[6]),
        ("m", cols[7]),
        ("m", cols[8]),
        ("m", cols[9]),
    ] {
        draw_right_text_line(image, unit, col, header_y + 26, MUTED, UNIT_SCALE);
    }
    hline(image, x, x + 1030, header_y + 46, LINE_DIM);

    for (i, parcel) in data.parcels.iter().enumerate() {
        let yy = row_y + i as i32 * 48;
        let label = if parcel.label == "Most-Unstable" {
            "Most-Unstbl"
        } else {
            parcel.label.as_str()
        };
        draw_text_line(image, label, x, yy, TEXT, BODY_SCALE);
        draw_right_text_line(
            image,
            &fmt_int(parcel.ecape),
            cols[1],
            yy,
            cape_color(parcel.ecape),
            3,
        );
        draw_right_text_line(image, &fmt_2f(parcel.ncape), cols[2], yy, TEXT, 3);
        draw_right_text_line(
            image,
            &fmt_int(parcel.cape),
            cols[3],
            yy,
            cape_color(parcel.cape),
            3,
        );
        draw_right_text_line(image, &fmt_int(parcel.cape_3km), cols[4], yy, TEXT, 3);
        draw_right_text_line(image, &fmt_int(parcel.cape_6km), cols[5], yy, TEXT, 3);
        draw_right_text_line(
            image,
            &fmt_int(parcel.cinh),
            cols[6],
            yy,
            cin_color(parcel.cinh),
            3,
        );
        draw_right_text_line(image, &fmt_int(parcel.lcl_m), cols[7], yy, TEXT, 3);
        draw_right_text_line(image, &fmt_int(parcel.lfc_m), cols[8], yy, TEXT, 3);
        draw_right_text_line(image, &fmt_int(parcel.el_m), cols[9], yy, TEXT, 3);
    }
}

fn draw_storm_motions(image: &mut RgbaImage, x: i32, y: i32, data: &TableData) {
    section_title(image, "STORM MOTIONS", x, y, 370);
    let mut yy = y + 56;
    for motion in &data.storm_motions {
        draw_text_line(image, &motion.label, x, yy, LABEL, 3);
        draw_right_text_line(
            image,
            &fmt_dir_spd(motion.direction, motion.speed),
            x + 320,
            yy - 2,
            TEXT,
            3,
        );
        yy += 46;
    }
    yy += 8;
    draw_text_line(image, "1km wind", x, yy, LABEL, 3);
    draw_right_text_line(
        image,
        &fmt_dir_spd(data.wind_1km_dir, data.wind_1km_spd),
        x + 320,
        yy - 2,
        GOOD,
        3,
    );
    yy += 46;
    draw_text_line(image, "6km wind", x, yy, LABEL, 3);
    draw_right_text_line(
        image,
        &fmt_dir_spd(data.wind_6km_dir, data.wind_6km_spd),
        x + 320,
        yy - 2,
        GOOD,
        3,
    );
}

fn draw_lapse_rates(image: &mut RgbaImage, x: i32, y: i32, data: &TableData) {
    section_title(image, "LAPSE RATES", x, y, 600);
    let mut yy = y + 56;
    for row in &data.lapse_rates {
        draw_text_line(image, &row.label, x, yy, LABEL, 3);
        draw_right_text_line(
            image,
            &format!("{} C/km", fmt_1f(row.value)),
            x + 260,
            yy - 2,
            lapse_color(row.value),
            3,
        );
        yy += 46;
    }
}

fn draw_shear(image: &mut RgbaImage, x: i32, y: i32, data: &TableData) {
    section_title(image, "SHEAR / HELICITY", x, y, 560);
    let header_y = y + 54;
    let row_y = header_y + 58;
    let cols = [x, x + 190, x + 270, x + 350, x + 430, x + 540];

    draw_text_line(image, "Layer", cols[0], header_y, LABEL, LABEL_SCALE);
    for (label, col) in [
        ("EHI", cols[1]),
        ("SRH", cols[2]),
        ("Shear", cols[3]),
        ("Mean", cols[4]),
        ("SRWind", cols[5]),
    ] {
        draw_right_text_line(image, label, col, header_y, LABEL, LABEL_SCALE);
    }
    for (unit, col) in [
        ("", cols[1]),
        ("m2/s2", cols[2]),
        ("kt", cols[3]),
        ("kt", cols[4]),
        ("deg/kt", cols[5]),
    ] {
        draw_right_text_line(image, unit, col, header_y + 26, MUTED, UNIT_SCALE);
    }
    hline(image, x, x + 560, header_y + 46, LINE_DIM);

    for (i, row) in data.shear_layers.iter().enumerate() {
        let yy = row_y + i as i32 * 48;
        draw_text_line(image, &row.label, cols[0], yy, TEXT, 3);
        draw_right_text_line(image, &fmt_1f(row.ehi), cols[1], yy - 2, TEXT, 3);
        draw_right_text_line(
            image,
            &fmt_int(row.srh),
            cols[2],
            yy - 2,
            srh_color(row.srh),
            3,
        );
        draw_right_text_line(
            image,
            &fmt_int(row.shear),
            cols[3],
            yy - 2,
            shear_color(row.shear),
            3,
        );
        draw_right_text_line(image, &fmt_int(row.mean_wind), cols[4], yy - 2, TEXT, 3);
        draw_right_text_line(
            image,
            &fmt_dir_spd(row.srw_dir, row.srw_spd),
            cols[5],
            yy - 2,
            TEXT,
            3,
        );
    }
}

fn draw_indices(image: &mut RgbaImage, x: i32, y: i32, data: &TableData) {
    section_title(image, "THERMODYNAMICS", x, y, 610);
    let left = [
        ("PWAT", fmt_unit(data.pw, "in", 2), TEXT),
        ("Mean MixR", fmt_unit(data.mean_mixr, "g/kg", 1), GOOD),
        ("Sfc RH", fmt_unit(data.sfc_rh, "%", 0), TEXT),
        ("Low RH", fmt_unit(data.low_rh, "%", 0), TEXT),
        ("Mid RH", fmt_unit(data.mid_rh, "%", 0), TEXT),
        ("DGZ RH", fmt_unit(data.dgz_rh, "%", 0), TEXT),
        ("Freezing", fmt_unit(data.freezing_level_m, "m", 0), TEXT),
        ("WB Zero", fmt_unit(data.wb_zero_m, "m", 0), GOOD),
        ("MU MPL", fmt_unit(data.mu_mpl_m, "m", 0), TEXT),
    ];
    let right = [
        ("3km Theta", fmt_unit(data.thetae_diff_3km, "K", 0), TEXT),
        ("LCL Temp", fmt_unit(data.lcl_temp_c, "C", 1), TEXT),
        ("ConvT", fmt_unit(data.conv_t, "C", 1), TEXT),
        ("MaxT", fmt_unit(data.max_t, "C", 1), TEXT),
        ("K Index", fmt_1f(data.k_index), TEXT),
        ("TotTots", fmt_1f(data.t_totals), TEXT),
        ("TEI", fmt_1f(data.tei), TEXT),
        ("TEHI", fmt_1f(data.tehi), stp_color(data.tehi)),
        ("TTS", fmt_1f(data.tts), stp_color(data.tts)),
    ];
    draw_key_columns(image, x, y + 54, &left, &right);
}

fn draw_composites(image: &mut RgbaImage, x: i32, y: i32, data: &TableData) {
    section_title(image, "COMPOSITES", x, y, 610);
    let left = [
        ("STP cin", fmt_1f(data.stp_cin), stp_color(data.stp_cin)),
        (
            "STP fixed",
            fmt_1f(data.stp_fixed),
            stp_color(data.stp_fixed),
        ),
        ("Supercell", fmt_1f(data.scp), scp_color(data.scp)),
        ("SHIP", fmt_1f(data.ship), ship_color(data.ship)),
        ("DCP", fmt_1f(data.dcp), TEXT),
        ("WNDG", fmt_1f(data.wndg), TEXT),
    ];
    let right = [
        ("VTP mod", fmt_1f(data.vtp_mod), stp_color(data.vtp_mod)),
        ("DCAPE", fmt_unit(data.dcape, "J/kg", 0), TEXT),
        ("DownT", fmt_unit(data.down_t, "C", 1), TEXT),
        ("MMP", fmt_2f(data.mmp), TEXT),
        ("ESP", fmt_1f(data.esp), TEXT),
        ("SigSvr", fmt_int(data.sig_svr), MUTED),
        ("LHP", fmt_1f(data.lhp), MUTED),
    ];
    draw_key_columns(image, x, y + 54, &left, &right);
}

fn draw_key_columns(
    image: &mut RgbaImage,
    x: i32,
    y: i32,
    left: &[(&str, String, Color)],
    right: &[(&str, String, Color)],
) {
    let rows = left.len().max(right.len());
    for i in 0..rows {
        let yy = y + i as i32 * 34;
        if let Some((label, value, color)) = left.get(i) {
            draw_text_line(image, label, x, yy, LABEL, 3);
            draw_right_text_line(image, value, x + 245, yy - 2, *color, 3);
        }
        if let Some((label, value, color)) = right.get(i) {
            draw_text_line(image, label, x + 310, yy, LABEL, 3);
            draw_right_text_line(image, value, x + 585, yy - 2, *color, 3);
        }
    }
}

fn section_title(image: &mut RgbaImage, title: &str, x: i32, y: i32, width: i32) {
    draw_text_line(image, title, x, y, LABEL, SECTION_SCALE);
    hline(image, x, x + width, y + 44, LINE);
}

fn build_table_data(
    profile: &SharprsProfile,
    p: &ComputedParams,
    ecape: &VerifiedEcapeParcels,
) -> TableData {
    let p_sfc = profile.sfc_pressure();
    let p500m = profile.pres_at_height(profile.to_msl(500.0));
    let p1km = profile.pres_at_height(profile.to_msl(1000.0));
    let p2km = profile.pres_at_height(profile.to_msl(2000.0));
    let p3km = profile.pres_at_height(profile.to_msl(3000.0));
    let p3500m = profile.pres_at_height(profile.to_msl(3500.0));
    let p6km = profile.pres_at_height(profile.to_msl(6000.0));
    let p12km = profile.pres_at_height(profile.to_msl(12000.0));

    let eff_bot_h = pressure_to_agl(profile, p.eff_inflow.0);
    let eff_top_h = pressure_to_agl(profile, p.eff_inflow.1);
    let lr03 = p
        .lr03
        .unwrap_or_else(|| lapse_rate_agl(profile, 0.0, 3000.0));
    let lr36 = p
        .lr36
        .unwrap_or_else(|| lapse_rate_agl(profile, 3000.0, 6000.0));

    let sfc_lcl_lr = if p.sfcpcl.lclhght.is_finite() && p.sfcpcl.lclhght > 1.0 {
        let direct = lapse_rate_agl(profile, 0.0, p.sfcpcl.lclhght);
        if direct.is_finite() {
            direct
        } else {
            let lcl_env_tmpc = profile.interp_tmpc(p.sfcpcl.lclpres);
            (profile.tmpc[profile.sfc] - lcl_env_tmpc) / p.sfcpcl.lclhght * 1000.0
        }
    } else {
        f64::NAN
    };

    let (rm_dir, rm_spd) = sharprs::profile::comp2vec(p.rstu, p.rstv);
    let (lm_dir, lm_spd) = sharprs::profile::comp2vec(p.lstu, p.lstv);
    let (cu_dir, cu_spd) = sharprs::profile::comp2vec(p.corfidi_up_u, p.corfidi_up_v);
    let (cd_dir, cd_spd) = sharprs::profile::comp2vec(p.corfidi_dn_u, p.corfidi_dn_v);
    let (_, lcl_temp_c) = cape::lcl(p_sfc, profile.tmpc[profile.sfc], profile.dwpc[profile.sfc]);
    let (dgz_bot, dgz_top) = indices::dgz(profile);
    let dgz_rh = indices::mean_relh(profile, Some(dgz_bot), Some(dgz_top)).unwrap_or(f64::NAN);
    let mean_wind_1_35_ms = mean_wind_mag(profile, p1km, p3500m) * 0.514_444;
    let wndg = composites::wndg(p.mlpcl.bplus, lr03, mean_wind_1_35_ms, p.mlpcl.bminus)
        .unwrap_or(f64::NAN);
    let shr06_mag = vector_mag(p.shr06.0, p.shr06.1);
    let mean06_mag = vector_mag(p.mean_wind_06.0, p.mean_wind_06.1);
    let dcp =
        composites::dcp(p.dcape.dcape, p.mupcl.bplus, shr06_mag, mean06_mag).unwrap_or(f64::NAN);
    let esp = composites::esp(p.mlpcl.b3km, lr03, p.mlpcl.bplus).unwrap_or(f64::NAN);
    let lr38 = indices::lapse_rate(profile, 3000.0, 8000.0, false).unwrap_or(f64::NAN);
    let mean_wind_3_12_ms = mean_wind_mag(profile, p3km, p12km) * 0.514_444;
    let mmp = {
        let max_bulk_shear = max_bulk_shear_0_1_to_6_10_mps(profile);
        if max_bulk_shear.is_finite() && lr38.is_finite() && mean_wind_3_12_ms.is_finite() {
            indices::coniglio(p.mupcl.bplus, max_bulk_shear, lr38, mean_wind_3_12_ms)
        } else {
            f64::NAN
        }
    };

    TableData {
        parcels: vec![
            parcel_row("Surface", &ecape.surface_based, &p.sfcpcl),
            parcel_row("Mixed-Layer", &ecape.mixed_layer, &p.mlpcl),
            parcel_row("Most-Unstable", &ecape.most_unstable, &p.mupcl),
        ],
        shear_layers: vec![
            shear_row(profile, p, "Sfc-500m", p_sfc, p500m, 0.0, 500.0),
            shear_row(profile, p, "Sfc-1km", p_sfc, p1km, 0.0, 1000.0),
            shear_row(
                profile,
                p,
                "Eff Inflow",
                p.eff_inflow.0,
                p.eff_inflow.1,
                eff_bot_h,
                eff_top_h,
            ),
            shear_row(profile, p, "Sfc-3km", p_sfc, p3km, 0.0, 3000.0),
            shear_row(profile, p, "1km-3km", p1km, p3km, 1000.0, 3000.0),
            shear_row(profile, p, "3km-6km", p3km, p6km, 3000.0, 6000.0),
            shear_row(profile, p, "Sfc-6km", p_sfc, p6km, 0.0, 6000.0),
            shear_row(profile, p, "Sfc-2km", p_sfc, p2km, 0.0, 2000.0),
        ],
        lapse_rates: vec![
            LapseRateRow {
                label: "Sfc-LCL".into(),
                value: sfc_lcl_lr,
            },
            LapseRateRow {
                label: "950-850 mb".into(),
                value: indices::lapse_rate(profile, 950.0, 850.0, true).unwrap_or(f64::NAN),
            },
            LapseRateRow {
                label: "Sfc-3km".into(),
                value: lr03,
            },
            LapseRateRow {
                label: "3km-6km".into(),
                value: lr36,
            },
            LapseRateRow {
                label: "850-500 mb".into(),
                value: p.lr85.unwrap_or(f64::NAN),
            },
            LapseRateRow {
                label: "700-500 mb".into(),
                value: p.lr75.unwrap_or(f64::NAN),
            },
        ],
        storm_motions: vec![
            StormMotionRow {
                label: "Bunkers RM".into(),
                direction: rm_dir,
                speed: rm_spd,
            },
            StormMotionRow {
                label: "Bunkers LM".into(),
                direction: lm_dir,
                speed: lm_spd,
            },
            StormMotionRow {
                label: "Corfidi Down".into(),
                direction: cd_dir,
                speed: cd_spd,
            },
            StormMotionRow {
                label: "Corfidi Up".into(),
                direction: cu_dir,
                speed: cu_spd,
            },
        ],
        pw: p.precip_water.unwrap_or(f64::NAN),
        mean_mixr: p.mean_mixr.unwrap_or(f64::NAN),
        sfc_rh: profile.relh.get(profile.sfc).copied().unwrap_or(f64::NAN),
        low_rh: p.mean_rh_low.unwrap_or(f64::NAN),
        mid_rh: p.mean_rh_mid.unwrap_or(f64::NAN),
        dgz_rh,
        freezing_level_m: p.frz_lvl.unwrap_or(f64::NAN),
        wb_zero_m: p.wb_zero.unwrap_or(f64::NAN),
        mu_mpl_m: p.mupcl.mplhght,
        thetae_diff_3km: p.tei.unwrap_or(f64::NAN),
        lcl_temp_c,
        conv_t: p.conv_t.unwrap_or(f64::NAN),
        max_t: p.max_temp.unwrap_or(f64::NAN),
        k_index: p.k_index.unwrap_or(f64::NAN),
        t_totals: p.t_totals.unwrap_or(f64::NAN),
        tei: p.tei.unwrap_or(f64::NAN),
        tehi: p.tehi.unwrap_or(f64::NAN),
        tts: p.tts.unwrap_or(f64::NAN),
        vtp_mod: p.vtp_mod.unwrap_or(f64::NAN),
        stp_cin: p.stp_cin.unwrap_or(f64::NAN),
        stp_fixed: p.stp_fixed.unwrap_or(f64::NAN),
        scp: p.scp.unwrap_or(f64::NAN),
        ship: p.ship.unwrap_or(f64::NAN),
        dcp,
        wndg,
        dcape: p.dcape.dcape,
        down_t: p.dcape.ttrace.last().copied().unwrap_or(f64::NAN),
        mmp,
        esp,
        sig_svr: f64::NAN,
        lhp: f64::NAN,
        wind_1km_dir: p.wind_1km.0,
        wind_1km_spd: p.wind_1km.1,
        wind_6km_dir: p.wind_6km.0,
        wind_6km_spd: p.wind_6km.1,
    }
}

fn parcel_row(
    label: &str,
    ecape: &VerifiedEcapeParcelParams,
    native: &cape::ParcelResult,
) -> ParcelRow {
    ParcelRow {
        label: label.to_string(),
        ecape: finite_or(ecape.ecape, f64::NAN),
        ncape: finite_or(ecape.ncape, f64::NAN),
        cape: finite_or(native.bplus, ecape.cape),
        cape_3km: finite_or(native.b3km, ecape.cape_3km),
        cape_6km: finite_or(native.b6km, ecape.cape_6km),
        cinh: finite_or(native.bminus, ecape.cinh),
        lcl_m: native.lclhght,
        lfc_m: native.lfchght,
        el_m: native.elhght,
    }
}

fn shear_row(
    profile: &SharprsProfile,
    p: &ComputedParams,
    label: &str,
    pbot: f64,
    ptop: f64,
    bottom_agl: f64,
    top_agl: f64,
) -> ShearRow {
    if !pbot.is_finite() || !ptop.is_finite() || !bottom_agl.is_finite() || !top_agl.is_finite() {
        return ShearRow::missing(label);
    }

    let srh = winds::helicity(profile, bottom_agl, top_agl, p.rstu, p.rstv, -1.0, false)
        .map(|value| value.0)
        .unwrap_or(f64::NAN);
    let ehi = composites::ehi(p.sfcpcl.bplus, srh).unwrap_or(f64::NAN);
    let (srw_dir, srw_spd) =
        if let Ok((su, sv)) = winds::sr_wind(profile, pbot, ptop, p.rstu, p.rstv, -1.0) {
            sharprs::profile::comp2vec(su, sv)
        } else {
            (f64::NAN, f64::NAN)
        };

    ShearRow {
        label: label.to_string(),
        ehi,
        srh,
        shear: shear_mag(profile, pbot, ptop),
        mean_wind: mean_wind_mag(profile, pbot, ptop),
        srw_dir,
        srw_spd,
    }
}

fn pressure_to_agl(profile: &SharprsProfile, pressure_hpa: f64) -> f64 {
    if !pressure_hpa.is_finite() {
        return f64::NAN;
    }
    let height = profile.interp_hght(pressure_hpa);
    if height.is_finite() {
        profile.to_agl(height)
    } else {
        f64::NAN
    }
}

fn shear_mag(profile: &SharprsProfile, pbot: f64, ptop: f64) -> f64 {
    winds::wind_shear(profile, pbot, ptop)
        .map(|(u, v)| vector_mag(u, v))
        .unwrap_or(f64::NAN)
}

fn mean_wind_mag(profile: &SharprsProfile, pbot: f64, ptop: f64) -> f64 {
    winds::mean_wind(profile, pbot, ptop, -1.0, 0.0, 0.0)
        .map(|(u, v)| vector_mag(u, v))
        .unwrap_or(f64::NAN)
}

fn vector_mag(u: f64, v: f64) -> f64 {
    (u * u + v * v).sqrt()
}

fn lapse_rate_agl(profile: &SharprsProfile, lower_agl: f64, upper_agl: f64) -> f64 {
    let lower_msl = profile.to_msl(lower_agl);
    let upper_msl = profile.to_msl(upper_agl);
    let t_lower = interp_profile_field_by_height(profile, lower_msl, &profile.tmpc);
    let t_upper = interp_profile_field_by_height(profile, upper_msl, &profile.tmpc);
    let dz = upper_msl - lower_msl;
    if t_lower.is_finite() && t_upper.is_finite() && dz.abs() > 1.0 {
        (t_upper - t_lower) / dz * -1000.0
    } else {
        f64::NAN
    }
}

fn interp_profile_field_by_height(profile: &SharprsProfile, target_msl: f64, field: &[f64]) -> f64 {
    if !target_msl.is_finite() || field.len() != profile.hght.len() {
        return f64::NAN;
    }
    for i in 0..profile.hght.len().saturating_sub(1) {
        let h0 = profile.hght[i];
        let h1 = profile.hght[i + 1];
        let v0 = field[i];
        let v1 = field[i + 1];
        if !h0.is_finite() || !h1.is_finite() || !v0.is_finite() || !v1.is_finite() {
            continue;
        }
        if (target_msl >= h0 && target_msl <= h1) || (target_msl <= h0 && target_msl >= h1) {
            let dh = h1 - h0;
            if dh.abs() < 1.0e-6 {
                return v0;
            }
            let frac = (target_msl - h0) / dh;
            return v0 + frac * (v1 - v0);
        }
    }
    f64::NAN
}

fn max_bulk_shear_0_1_to_6_10_mps(profile: &SharprsProfile) -> f64 {
    let low = wind_indices_in_layer(profile, 0.0, 1000.0);
    let high = wind_indices_in_layer(profile, 6000.0, 10_000.0);
    let mut max_shear = f64::NAN;
    for &i in &low {
        for &j in &high {
            let du = profile.u[j] - profile.u[i];
            let dv = profile.v[j] - profile.v[i];
            if du.is_finite() && dv.is_finite() {
                let shear = vector_mag(du, dv) * 0.514_444;
                if !max_shear.is_finite() || shear > max_shear {
                    max_shear = shear;
                }
            }
        }
    }
    max_shear
}

fn wind_indices_in_layer(profile: &SharprsProfile, bottom_agl: f64, top_agl: f64) -> Vec<usize> {
    profile
        .hght
        .iter()
        .enumerate()
        .filter_map(|(index, &height_msl)| {
            let agl = profile.to_agl(height_msl);
            if agl >= bottom_agl
                && agl <= top_agl
                && profile.u[index].is_finite()
                && profile.v[index].is_finite()
            {
                Some(index)
            } else {
                None
            }
        })
        .collect()
}

fn finite_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() { value } else { fallback }
}

fn fmt_int(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.0}")
    } else {
        "--".to_string()
    }
}

fn fmt_1f(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.1}")
    } else {
        "--".to_string()
    }
}

fn fmt_2f(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.2}")
    } else {
        "--".to_string()
    }
}

fn fmt_unit(value: f64, unit: &str, precision: usize) -> String {
    if !value.is_finite() {
        return "--".to_string();
    }
    let number = match precision {
        0 => format!("{value:.0}"),
        1 => format!("{value:.1}"),
        _ => format!("{value:.2}"),
    };
    format!("{number} {unit}")
}

fn fmt_dir_spd(direction: f64, speed: f64) -> String {
    if direction.is_finite() && speed.is_finite() {
        format!("{direction:.0}/{speed:.0}")
    } else {
        "--".to_string()
    }
}

fn cape_color(value: f64) -> Color {
    if !value.is_finite() {
        return MUTED;
    }
    if value >= 4000.0 {
        DANGER
    } else if value >= 2500.0 {
        ORANGE
    } else if value >= 1000.0 {
        WATCH
    } else {
        TEXT
    }
}

fn cin_color(value: f64) -> Color {
    if !value.is_finite() {
        MUTED
    } else if value <= -100.0 {
        DANGER
    } else if value <= -50.0 {
        ORANGE
    } else {
        GOOD
    }
}

fn lapse_color(value: f64) -> Color {
    if !value.is_finite() {
        MUTED
    } else if value >= 8.5 {
        DANGER
    } else if value >= 7.5 {
        ORANGE
    } else if value >= 6.5 {
        WATCH
    } else {
        TEXT
    }
}

fn srh_color(value: f64) -> Color {
    if !value.is_finite() {
        MUTED
    } else if value >= 300.0 {
        DANGER
    } else if value >= 150.0 {
        ORANGE
    } else if value >= 75.0 {
        WATCH
    } else {
        TEXT
    }
}

fn shear_color(value: f64) -> Color {
    if !value.is_finite() {
        MUTED
    } else if value >= 60.0 {
        DANGER
    } else if value >= 40.0 {
        WATCH
    } else {
        TEXT
    }
}

fn stp_color(value: f64) -> Color {
    if !value.is_finite() {
        MUTED
    } else if value >= 4.0 {
        DANGER
    } else if value >= 2.0 {
        ORANGE
    } else if value >= 1.0 {
        WATCH
    } else {
        TEXT
    }
}

fn scp_color(value: f64) -> Color {
    if !value.is_finite() {
        MUTED
    } else if value >= 8.0 {
        DANGER
    } else if value >= 4.0 {
        ORANGE
    } else if value >= 1.0 {
        WATCH
    } else {
        TEXT
    }
}

fn ship_color(value: f64) -> Color {
    if !value.is_finite() {
        MUTED
    } else if value >= 2.0 {
        DANGER
    } else if value >= 1.0 {
        WATCH
    } else {
        TEXT
    }
}

fn fill_rect(image: &mut RgbaImage, x: i32, y: i32, w: i32, h: i32, color: Color) {
    let x0 = x.max(0) as u32;
    let y0 = y.max(0) as u32;
    let x1 = (x + w).max(0).min(image.width() as i32) as u32;
    let y1 = (y + h).max(0).min(image.height() as i32) as u32;
    let rgba = color_to_rgba(color);
    for py in y0..y1 {
        for px in x0..x1 {
            image.put_pixel(px, py, rgba);
        }
    }
}

fn hline(image: &mut RgbaImage, x0: i32, x1: i32, y: i32, color: Color) {
    if y < 0 || y >= image.height() as i32 {
        return;
    }
    let start = x0.min(x1).max(0) as u32;
    let end = x0.max(x1).min(image.width() as i32 - 1) as u32;
    let rgba = color_to_rgba(color);
    for x in start..=end {
        image.put_pixel(x, y as u32, rgba);
    }
}

fn vline(image: &mut RgbaImage, x: i32, y0: i32, y1: i32, color: Color) {
    if x < 0 || x >= image.width() as i32 {
        return;
    }
    let start = y0.min(y1).max(0) as u32;
    let end = y0.max(y1).min(image.height() as i32 - 1) as u32;
    let rgba = color_to_rgba(color);
    for y in start..=end {
        image.put_pixel(x as u32, y, rgba);
    }
}

fn draw_rect_border(
    image: &mut RgbaImage,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    color: Color,
    width: i32,
) {
    for offset in 0..width.max(1) {
        hline(image, x + offset, x + w - 1 - offset, y + offset, color);
        hline(
            image,
            x + offset,
            x + w - 1 - offset,
            y + h - 1 - offset,
            color,
        );
        vline(image, x + offset, y + offset, y + h - 1 - offset, color);
        vline(
            image,
            x + w - 1 - offset,
            y + offset,
            y + h - 1 - offset,
            color,
        );
    }
}

fn draw_line(image: &mut RgbaImage, x0: i32, y0: i32, x1: i32, y1: i32, color: Color) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    let rgba = color_to_rgba(color);

    loop {
        if x >= 0 && x < image.width() as i32 && y >= 0 && y < image.height() as i32 {
            image.put_pixel(x as u32, y as u32, rgba);
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn draw_line_width(
    image: &mut RgbaImage,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: Color,
    width: i32,
) {
    let half = (width.max(1) - 1) / 2;
    for offset in -half..=half {
        draw_line(image, x0 + offset, y0, x1 + offset, y1, color);
        if offset != 0 {
            draw_line(image, x0, y0 + offset, x1, y1 + offset, color);
        }
    }
}

fn draw_circle(image: &mut RgbaImage, cx: i32, cy: i32, r: i32, color: Color) {
    let mut x = r.max(0);
    let mut y = 0;
    let mut err = 0;
    let rgba = color_to_rgba(color);
    while x >= y {
        for (px, py) in [
            (cx + x, cy + y),
            (cx + y, cy + x),
            (cx - y, cy + x),
            (cx - x, cy + y),
            (cx - x, cy - y),
            (cx - y, cy - x),
            (cx + y, cy - x),
            (cx + x, cy - y),
        ] {
            if px >= 0 && px < image.width() as i32 && py >= 0 && py < image.height() as i32 {
                image.put_pixel(px as u32, py as u32, rgba);
            }
        }
        y += 1;
        if err <= 0 {
            err += 2 * y + 1;
        }
        if err > 0 {
            x -= 1;
            err -= 2 * x + 1;
        }
    }
}

fn fill_circle(image: &mut RgbaImage, cx: i32, cy: i32, r: i32, color: Color) {
    let radius = r.max(0);
    let radius_sq = radius * radius;
    let rgba = color_to_rgba(color);
    for y in -radius..=radius {
        for x in -radius..=radius {
            if x * x + y * y > radius_sq {
                continue;
            }
            let px = cx + x;
            let py = cy + y;
            if px >= 0 && px < image.width() as i32 && py >= 0 && py < image.height() as i32 {
                image.put_pixel(px as u32, py as u32, rgba);
            }
        }
    }
}

fn color_to_rgba(color: Color) -> Rgba<u8> {
    Rgba([color.r, color.g, color.b, color.a])
}

fn encode_png(image: RgbaImage) -> Result<Vec<u8>, image::ImageError> {
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image).write_to(&mut bytes, ImageFormat::Png)?;
    Ok(bytes.into_inner())
}

struct TableData {
    parcels: Vec<ParcelRow>,
    shear_layers: Vec<ShearRow>,
    lapse_rates: Vec<LapseRateRow>,
    storm_motions: Vec<StormMotionRow>,
    pw: f64,
    mean_mixr: f64,
    sfc_rh: f64,
    low_rh: f64,
    mid_rh: f64,
    dgz_rh: f64,
    freezing_level_m: f64,
    wb_zero_m: f64,
    mu_mpl_m: f64,
    thetae_diff_3km: f64,
    lcl_temp_c: f64,
    conv_t: f64,
    max_t: f64,
    k_index: f64,
    t_totals: f64,
    tei: f64,
    tehi: f64,
    tts: f64,
    vtp_mod: f64,
    stp_cin: f64,
    stp_fixed: f64,
    scp: f64,
    ship: f64,
    dcp: f64,
    wndg: f64,
    dcape: f64,
    down_t: f64,
    mmp: f64,
    esp: f64,
    sig_svr: f64,
    lhp: f64,
    wind_1km_dir: f64,
    wind_1km_spd: f64,
    wind_6km_dir: f64,
    wind_6km_spd: f64,
}

struct ParcelRow {
    label: String,
    ecape: f64,
    ncape: f64,
    cape: f64,
    cape_3km: f64,
    cape_6km: f64,
    cinh: f64,
    lcl_m: f64,
    lfc_m: f64,
    el_m: f64,
}

struct ShearRow {
    label: String,
    ehi: f64,
    srh: f64,
    shear: f64,
    mean_wind: f64,
    srw_dir: f64,
    srw_spd: f64,
}

impl ShearRow {
    fn missing(label: &str) -> Self {
        Self {
            label: label.to_string(),
            ehi: f64::NAN,
            srh: f64::NAN,
            shear: f64::NAN,
            mean_wind: f64::NAN,
            srw_dir: f64::NAN,
            srw_spd: f64::NAN,
        }
    }
}

struct LapseRateRow {
    label: String,
    value: f64,
}

struct StormMotionRow {
    label: String,
    direction: f64,
    speed: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parcel_row_uses_native_parcel_values() {
        let ecape = VerifiedEcapeParcelParams {
            ecape: 1151.0,
            ncape: 0.17,
            cape: 1964.0,
            cinh: 0.0,
            cape_3km: 220.0,
            cape_6km: 828.0,
            lfc_m: 0.0,
            el_m: 9999.0,
        };
        let native = cape::ParcelResult {
            bplus: 1361.0,
            bminus: 0.0,
            b3km: 180.0,
            b6km: 700.0,
            lclhght: 343.0,
            lfchght: 343.0,
            elhght: 11538.0,
            ..cape::ParcelResult::default()
        };

        let row = parcel_row("Surface", &ecape, &native);

        assert_eq!(row.ecape, 1151.0);
        assert_eq!(row.ncape, 0.17);
        assert_eq!(row.cape, 1361.0);
        assert_eq!(row.cape_3km, 180.0);
        assert_eq!(row.cape_6km, 700.0);
        assert_eq!(row.cinh, 0.0);
        assert_eq!(row.lcl_m, 343.0);
        assert_eq!(row.lfc_m, 343.0);
        assert_eq!(row.el_m, 11538.0);
    }
}
