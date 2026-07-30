use crate::color::Rgba;
use image::RgbaImage;

pub fn blend_pixel(img: &mut RgbaImage, x: i32, y: i32, color: Rgba) {
    if x < 0 || y < 0 || (x as u32) >= img.width() || (y as u32) >= img.height() {
        return;
    }

    if color.a == 255 {
        img.put_pixel(x as u32, y as u32, color.to_image_rgba());
        return;
    }
    if color.a == 0 {
        return;
    }

    let dst = img.get_pixel(x as u32, y as u32).0;
    let alpha = color.a as f64 / 255.0;
    let inv = 1.0 - alpha;
    let blended = image::Rgba([
        (color.r as f64 * alpha + dst[0] as f64 * inv).round() as u8,
        (color.g as f64 * alpha + dst[1] as f64 * inv).round() as u8,
        (color.b as f64 * alpha + dst[2] as f64 * inv).round() as u8,
        255,
    ]);
    img.put_pixel(x as u32, y as u32, blended);
}

fn blend_pixel_coverage(img: &mut RgbaImage, x: i32, y: i32, color: Rgba, coverage: f64) {
    if coverage <= 0.0 || color.a == 0 {
        return;
    }
    let scaled_alpha = ((color.a as f64) * coverage.clamp(0.0, 1.0)).round() as u8;
    if scaled_alpha == 0 {
        return;
    }
    blend_pixel(
        img,
        x,
        y,
        Rgba {
            a: scaled_alpha,
            ..color
        },
    );
}

pub fn draw_circle_fill_aa(img: &mut RgbaImage, cx: f64, cy: f64, radius: f64, color: Rgba) {
    if !cx.is_finite() || !cy.is_finite() || !radius.is_finite() || radius <= 0.0 || color.a == 0 {
        return;
    }

    let radius = radius.max(0.0);
    let bounds = radius + 1.0;
    let min_x = (cx - bounds).floor() as i32;
    let max_x = (cx + bounds).ceil() as i32;
    let min_y = (cy - bounds).floor() as i32;
    let max_y = (cy + bounds).ceil() as i32;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let dx = (x as f64 + 0.5) - cx;
            let dy = (y as f64 + 0.5) - cy;
            let distance = (dx * dx + dy * dy).sqrt();
            let coverage = (radius + 0.5 - distance).clamp(0.0, 1.0);
            blend_pixel_coverage(img, x, y, color, coverage);
        }
    }
}

pub fn draw_circle_stroke_aa(
    img: &mut RgbaImage,
    cx: f64,
    cy: f64,
    radius: f64,
    color: Rgba,
    width: u32,
) {
    if !cx.is_finite()
        || !cy.is_finite()
        || !radius.is_finite()
        || radius <= 0.0
        || color.a == 0
        || width == 0
    {
        return;
    }

    let half_width = width.max(1) as f64 * 0.5;
    let bounds = radius + half_width + 1.0;
    let min_x = (cx - bounds).floor() as i32;
    let max_x = (cx + bounds).ceil() as i32;
    let min_y = (cy - bounds).floor() as i32;
    let max_y = (cy + bounds).ceil() as i32;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let dx = (x as f64 + 0.5) - cx;
            let dy = (y as f64 + 0.5) - cy;
            let distance = (dx * dx + dy * dy).sqrt();
            let coverage = (half_width + 0.5 - (distance - radius).abs()).clamp(0.0, 1.0);
            blend_pixel_coverage(img, x, y, color, coverage);
        }
    }
}

fn distance_to_segment(px: f64, py: f64, x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len_sq = dx * dx + dy * dy;
    if len_sq <= 1e-12 {
        let ox = px - x0;
        let oy = py - y0;
        return (ox * ox + oy * oy).sqrt();
    }

    let t = (((px - x0) * dx + (py - y0) * dy) / len_sq).clamp(0.0, 1.0);
    let proj_x = x0 + t * dx;
    let proj_y = y0 + t * dy;
    let ox = px - proj_x;
    let oy = py - proj_y;
    (ox * ox + oy * oy).sqrt()
}

fn stroke_coverage(distance: f64, width: u32) -> f64 {
    let half_width = width.max(1) as f64 * 0.5;
    (half_width + 0.5 - distance).clamp(0.0, 1.0)
}

fn draw_line_aa_kernel(
    img: &mut RgbaImage,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    color: Rgba,
    width: u32,
) {
    if !x0.is_finite() || !y0.is_finite() || !x1.is_finite() || !y1.is_finite() {
        return;
    }

    let radius = width.max(1) as f64 * 0.5 + 1.0;
    let min_x = (x0.min(x1) - radius).floor() as i32;
    let max_x = (x0.max(x1) + radius).ceil() as i32;
    let min_y = (y0.min(y1) - radius).floor() as i32;
    let max_y = (y0.max(y1) + radius).ceil() as i32;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px = x as f64 + 0.5;
            let py = y as f64 + 0.5;
            let coverage = stroke_coverage(distance_to_segment(px, py, x0, y0, x1, y1), width);
            blend_pixel_coverage(img, x, y, color, coverage);
        }
    }
}

pub fn draw_line_aa_width(
    img: &mut RgbaImage,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    color: Rgba,
    width: u32,
) {
    draw_line_aa_kernel(img, x0, y0, x1, y1, color, width.max(1));
}

pub fn draw_plus_marker_aa(
    img: &mut RgbaImage,
    cx: f64,
    cy: f64,
    radius: f64,
    color: Rgba,
    width: u32,
) {
    if !cx.is_finite() || !cy.is_finite() || !radius.is_finite() || radius <= 0.0 {
        return;
    }
    draw_line_aa_width(img, cx - radius, cy, cx + radius, cy, color, width);
    draw_line_aa_width(img, cx, cy - radius, cx, cy + radius, color, width);
}

pub fn draw_cross_marker_aa(
    img: &mut RgbaImage,
    cx: f64,
    cy: f64,
    radius: f64,
    color: Rgba,
    width: u32,
) {
    if !cx.is_finite() || !cy.is_finite() || !radius.is_finite() || radius <= 0.0 {
        return;
    }
    let r = radius / 2.0_f64.sqrt();
    draw_line_aa_width(img, cx - r, cy - r, cx + r, cy + r, color, width);
    draw_line_aa_width(img, cx - r, cy + r, cx + r, cy - r, color, width);
}

pub fn draw_polyline_aa(img: &mut RgbaImage, points: &[(f64, f64)], color: Rgba, width: u32) {
    if points.len() < 2 {
        return;
    }
    for segment in points.windows(2) {
        let (x0, y0) = segment[0];
        let (x1, y1) = segment[1];
        draw_line_aa_width(img, x0, y0, x1, y1, color, width);
    }
}

/// Fill a polygon (optionally with holes) using an even-odd scanline rule.
///
/// `rings` is a list of closed rings in pixel coordinates — the first ring is
/// the outer boundary, any additional rings punch holes out of it. Rings are
/// auto-closed (the last vertex does not need to repeat the first).
///
/// Clipped to the image bounds. Uses alpha-blended pixel writes so fills with
/// partial alpha composite correctly over existing pixels.
/// Scanline-fills `rings` with even-odd winding, clipped to `clip` (inclusive
/// pixel rect: `(x0, y0, x1, y1)`) when supplied, otherwise clipped to the full
/// image. Callers drawing into a map panel must pass the panel rect so that
/// global polygons (world oceans / continents) don't bleed into the margins.
pub fn fill_polygon(
    img: &mut RgbaImage,
    rings: &[Vec<(f64, f64)>],
    color: Rgba,
    clip: Option<(i32, i32, i32, i32)>,
) {
    if rings.is_empty() || color.a == 0 {
        return;
    }

    let img_w = img.width() as i32;
    let img_h = img.height() as i32;
    let (cx0, cy0, cx1, cy1) = match clip {
        Some((x0, y0, x1, y1)) => (x0.max(0), y0.max(0), x1.min(img_w - 1), y1.min(img_h - 1)),
        None => (0, 0, img_w - 1, img_h - 1),
    };
    if cx1 < cx0 || cy1 < cy0 {
        return;
    }

    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for ring in rings {
        for &(_, y) in ring {
            if y.is_finite() {
                y_min = y_min.min(y);
                y_max = y_max.max(y);
            }
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() || y_max < cy0 as f64 {
        return;
    }
    let y0 = y_min.floor().max(cy0 as f64) as i32;
    let y1 = (y_max.ceil() as i32).min(cy1);
    if y1 < y0 {
        return;
    }

    // Pre-extract edges once. Storing (y_min, y_max, x_at_y_min, dx_per_dy)
    // lets the scanline loop skip edges that don't span it.
    #[derive(Clone)]
    struct Edge {
        y_min: f64,
        y_max: f64,
        x: f64,
        dx: f64,
    }
    let mut edges: Vec<Edge> = Vec::new();
    for ring in rings {
        let n = ring.len();
        if n < 2 {
            continue;
        }
        for i in 0..n {
            let (ax, ay) = ring[i];
            let (bx, by) = ring[(i + 1) % n];
            if !ax.is_finite() || !ay.is_finite() || !bx.is_finite() || !by.is_finite() {
                continue;
            }
            if (ay - by).abs() < 1e-9 {
                continue; // horizontal edges contribute nothing to even-odd
            }
            let (lo_y, hi_y, lo_x, hi_x) = if ay < by {
                (ay, by, ax, bx)
            } else {
                (by, ay, bx, ax)
            };
            let dx = (hi_x - lo_x) / (hi_y - lo_y);
            edges.push(Edge {
                y_min: lo_y,
                y_max: hi_y,
                x: lo_x,
                dx,
            });
        }
    }
    if edges.is_empty() {
        return;
    }

    // Scanline loop. At pixel center (y + 0.5), collect edge x-intersections
    // for edges that straddle the scanline (using half-open [y_min, y_max)
    // avoids double-counting shared endpoints).
    let mut xs: Vec<f64> = Vec::with_capacity(edges.len());
    let opaque_fill = color.a == 255;
    let opaque_rgba = color.to_image_rgba();
    for y in y0..=y1 {
        let yf = y as f64 + 0.5;
        xs.clear();
        for edge in &edges {
            if yf >= edge.y_min && yf < edge.y_max {
                xs.push(edge.x + (yf - edge.y_min) * edge.dx);
            }
        }
        if xs.len() < 2 {
            continue;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut i = 0;
        while i + 1 < xs.len() {
            let xa = xs[i].max(cx0 as f64).ceil() as i32;
            let xb = xs[i + 1].min(cx1 as f64).floor() as i32;
            if xb >= xa {
                if opaque_fill {
                    for x in xa..=xb {
                        img.put_pixel(x as u32, y as u32, opaque_rgba);
                    }
                } else {
                    for x in xa..=xb {
                        blend_pixel(img, x, y, color);
                    }
                }
            }
            i += 2;
        }
    }
}

pub fn draw_wind_barb(
    img: &mut RgbaImage,
    x_tip: f64,
    y_tip: f64,
    u: f64,
    v: f64,
    color: Rgba,
    shaft_len: f64,
    width: u32,
) {
    if !u.is_finite() || !v.is_finite() {
        return;
    }

    let speed = (u * u + v * v).sqrt();
    if speed < 2.5 {
        draw_circle_fill_aa(img, x_tip, y_tip, 2.0, color);
        return;
    }

    // Screen-space unit vector from barb tip toward the tail.
    let tail_dx = -u / speed;
    let tail_dy = v / speed;
    // Matplotlib's default barb side lands on the counterclockwise
    // perpendicular in screen space for the tip-anchored shaft.
    let perp_dx = -tail_dy;
    let perp_dy = tail_dx;

    let tail_x = x_tip + tail_dx * shaft_len;
    let tail_y = y_tip + tail_dy * shaft_len;
    draw_line_aa_width(img, tail_x, tail_y, x_tip, y_tip, color, width);

    let mut remaining = ((speed + 2.5) / 5.0).floor() as i32 * 5;
    let mut offset = shaft_len;
    let spacing = (shaft_len * 0.16).max(2.0);
    let full_height = shaft_len * 0.40;
    let full_width = shaft_len * 0.25;

    while remaining >= 50 {
        draw_barb_flag(
            img,
            x_tip,
            y_tip,
            tail_dx,
            tail_dy,
            perp_dx,
            perp_dy,
            offset,
            full_height,
            full_width,
            color,
            width,
        );
        remaining -= 50;
        offset -= full_width + spacing;
    }

    while remaining >= 10 {
        draw_barb_segment(
            img,
            x_tip,
            y_tip,
            tail_dx,
            tail_dy,
            perp_dx,
            perp_dy,
            offset,
            full_height,
            full_width * 0.5,
            color,
            width,
        );
        remaining -= 10;
        offset -= spacing;
    }

    if remaining >= 5 {
        if (offset - shaft_len).abs() < 1e-6 {
            offset -= 1.5 * spacing;
        }
        draw_barb_segment(
            img,
            x_tip,
            y_tip,
            tail_dx,
            tail_dy,
            perp_dx,
            perp_dy,
            offset,
            full_height * 0.5,
            full_width * 0.25,
            color,
            width,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anti_aliased_line_blends_neighbor_pixels() {
        let mut img = RgbaImage::from_pixel(8, 8, image::Rgba([255, 255, 255, 255]));
        draw_line_aa_width(&mut img, 1.0, 1.0, 6.0, 4.0, Rgba::BLACK, 1);

        let mut blended_neighbor_found = false;
        for pixel in img.pixels() {
            let rgb = &pixel.0[..3];
            if *rgb != [255, 255, 255] && *rgb != [0, 0, 0] {
                blended_neighbor_found = true;
                break;
            }
        }

        assert!(blended_neighbor_found);
    }

    #[test]
    fn wide_anti_aliased_line_blends_neighbor_pixels() {
        let mut img = RgbaImage::from_pixel(16, 16, image::Rgba([255, 255, 255, 255]));
        draw_line_aa_width(&mut img, 2.0, 3.0, 13.0, 10.0, Rgba::BLACK, 6);

        let mut blended_neighbor_found = false;
        for pixel in img.pixels() {
            let rgb = &pixel.0[..3];
            if *rgb != [255, 255, 255] && *rgb != [0, 0, 0] {
                blended_neighbor_found = true;
                break;
            }
        }

        assert!(blended_neighbor_found);
    }

    #[test]
    fn wind_barb_blends_neighbor_pixels() {
        let mut img = RgbaImage::from_pixel(48, 48, image::Rgba([255, 255, 255, 255]));
        draw_wind_barb(&mut img, 24.0, 24.0, 20.0, -10.0, Rgba::BLACK, 16.0, 2);

        let mut blended_neighbor_found = false;
        for pixel in img.pixels() {
            let rgb = &pixel.0[..3];
            if *rgb != [255, 255, 255] && *rgb != [0, 0, 0] {
                blended_neighbor_found = true;
                break;
            }
        }

        assert!(blended_neighbor_found);
    }

    #[test]
    fn anti_aliased_circle_fill_blends_edge_pixels() {
        let mut img = RgbaImage::from_pixel(24, 24, image::Rgba([255, 255, 255, 255]));
        draw_circle_fill_aa(&mut img, 12.0, 12.0, 5.5, Rgba::BLACK);

        let blended = img
            .pixels()
            .any(|pixel| pixel.0[..3] != [255, 255, 255] && pixel.0[..3] != [0, 0, 0]);
        assert!(blended);
    }

    #[test]
    fn anti_aliased_circle_stroke_blends_edge_pixels() {
        let mut img = RgbaImage::from_pixel(24, 24, image::Rgba([255, 255, 255, 255]));
        draw_circle_stroke_aa(&mut img, 12.0, 12.0, 6.0, Rgba::BLACK, 2);

        let blended = img
            .pixels()
            .any(|pixel| pixel.0[..3] != [255, 255, 255] && pixel.0[..3] != [0, 0, 0]);
        assert!(blended);
    }
}

fn draw_barb_segment(
    img: &mut RgbaImage,
    x_tip: f64,
    y_tip: f64,
    tail_dx: f64,
    tail_dy: f64,
    perp_dx: f64,
    perp_dy: f64,
    offset: f64,
    height: f64,
    along_tail: f64,
    color: Rgba,
    width: u32,
) {
    let base_x = x_tip + tail_dx * offset;
    let base_y = y_tip + tail_dy * offset;
    let feather_x = base_x + perp_dx * height + tail_dx * along_tail;
    let feather_y = base_y + perp_dy * height + tail_dy * along_tail;
    draw_line_aa_width(img, base_x, base_y, feather_x, feather_y, color, width);
}

fn draw_barb_flag(
    img: &mut RgbaImage,
    x_tip: f64,
    y_tip: f64,
    tail_dx: f64,
    tail_dy: f64,
    perp_dx: f64,
    perp_dy: f64,
    offset: f64,
    height: f64,
    width_along: f64,
    color: Rgba,
    width: u32,
) {
    let base_x = x_tip + tail_dx * offset;
    let base_y = y_tip + tail_dy * offset;
    let flag_tip_x = base_x + perp_dx * height - tail_dx * (width_along * 0.5);
    let flag_tip_y = base_y + perp_dy * height - tail_dy * (width_along * 0.5);
    let flag_tail_x = base_x - tail_dx * width_along;
    let flag_tail_y = base_y - tail_dy * width_along;
    draw_line_aa_width(
        img,
        base_x,
        base_y,
        flag_tip_x,
        flag_tip_y,
        color,
        width + 1,
    );
    draw_line_aa_width(
        img,
        flag_tip_x,
        flag_tip_y,
        flag_tail_x,
        flag_tail_y,
        color,
        width + 1,
    );
    draw_line_aa_width(
        img,
        flag_tail_x,
        flag_tail_y,
        base_x,
        base_y,
        color,
        width + 1,
    );
}
