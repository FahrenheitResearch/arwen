//! Station model plot rendering.
//!
//! Renders standard meteorological station models onto an RGBA pixel buffer.
//! Each station shows wind barbs, sky cover circle, temperature, dewpoint,
//! pressure, and other parameters in the traditional layout:
//!
//! ```text
//!         TT     (temperature, upper-left)
//!    dd
//!   ====> ww    (wind barb from left, weather symbol center)
//!         Td     (dewpoint, lower-left)
//!    PPP         (pressure, upper-right)
//! ```

// colormap module available for future weather symbol coloring
#[allow(unused_imports)]
use super::colormap;

// ═══════════════════════════════════════════════════════════
// Data types
// ═══════════════════════════════════════════════════════════

/// A single surface weather observation for station model plotting.
#[derive(Debug, Clone)]
pub struct StationObs {
    pub lat: f64,
    pub lon: f64,
    /// Temperature in display units (°F or °C)
    pub temperature: Option<f64>,
    /// Dewpoint in display units
    pub dewpoint: Option<f64>,
    /// Wind speed in knots
    pub wind_speed: Option<f64>,
    /// Wind direction in meteorological degrees (0=N, 90=E, 180=S, 270=W)
    pub wind_direction: Option<f64>,
    /// Sea-level pressure in hPa (last 3 digits plotted)
    pub pressure: Option<f64>,
    /// Sky cover in oktas (0=clear, 4=half, 8=overcast)
    pub sky_cover: Option<u8>,
    /// WMO present weather code (0-99)
    pub weather: Option<u8>,
    /// Visibility in statute miles
    pub visibility: Option<f64>,
    /// 3-hour pressure tendency in hPa
    pub pressure_tendency: Option<f64>,
}

/// Configuration for station model plot rendering.
#[derive(Debug, Clone)]
pub struct StationPlotConfig {
    /// Output image width in pixels
    pub width: u32,
    /// Output image height in pixels
    pub height: u32,
    /// Western boundary longitude
    pub lon_min: f64,
    /// Eastern boundary longitude
    pub lon_max: f64,
    /// Southern boundary latitude
    pub lat_min: f64,
    /// Northern boundary latitude
    pub lat_max: f64,
    /// Pixel footprint per station model (default 60)
    pub station_size: u32,
    /// Font size for text labels (default 12)
    pub font_size: u8,
    /// Background color as (R, G, B, A)
    pub bg_color: (u8, u8, u8, u8),
    /// Skip stations closer than this many degrees (default 0.5)
    pub thinning_radius: f64,
}

impl Default for StationPlotConfig {
    fn default() -> Self {
        Self {
            width: 1200,
            height: 900,
            lon_min: -130.0,
            lon_max: -60.0,
            lat_min: 20.0,
            lat_max: 55.0,
            station_size: 60,
            font_size: 12,
            bg_color: (255, 255, 255, 255),
            thinning_radius: 0.5,
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Bitmap font: 5x7 pixels for digits, minus, decimal, slash, letters
// ═══════════════════════════════════════════════════════════

/// Each glyph is 5 columns x 7 rows, stored as 7 bytes (each byte = 5 LSBs = columns).
/// Bit 4 = leftmost column, bit 0 = rightmost column.
const FONT_5X7: [(char, [u8; 7]); 50] = [
    ('0', [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110]),
    ('1', [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
    ('2', [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111]),
    ('3', [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110]),
    ('4', [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010]),
    ('5', [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110]),
    ('6', [0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110]),
    ('7', [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000]),
    ('8', [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110]),
    ('9', [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110]),
    ('-', [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000]),
    ('.', [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100]),
    ('/', [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000]),
    (' ', [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000]),
    ('A', [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001]),
    ('B', [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110]),
    ('C', [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110]),
    ('D', [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110]),
    ('E', [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111]),
    ('F', [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000]),
    ('G', [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110]),
    ('H', [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001]),
    ('I', [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
    ('J', [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100]),
    ('K', [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001]),
    ('L', [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111]),
    ('M', [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001]),
    ('N', [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001]),
    ('O', [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110]),
    ('P', [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000]),
    ('Q', [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101]),
    ('R', [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001]),
    ('S', [0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110]),
    ('T', [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100]),
    ('U', [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110]),
    ('V', [0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b01010, 0b00100]),
    ('W', [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001]),
    ('X', [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001]),
    ('Y', [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100]),
    ('Z', [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111]),
    // lowercase mapped to uppercase rendering
    ('a', [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001]),
    ('b', [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110]),
    ('c', [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110]),
    ('d', [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110]),
    ('e', [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111]),
    ('f', [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000]),
    ('k', [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001]),
    ('t', [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100]),
    ('n', [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001]),
    ('s', [0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110]),
];

/// Look up a glyph bitmap. Returns None for unknown characters.
fn glyph(ch: char) -> Option<[u8; 7]> {
    for &(c, data) in &FONT_5X7 {
        if c == ch {
            return Some(data);
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════
// Drawing primitives on an RGBA buffer
// ═══════════════════════════════════════════════════════════

/// Set a pixel in the RGBA buffer if in bounds.
#[inline]
fn put_pixel(buf: &mut [u8], w: u32, h: u32, x: i32, y: i32, r: u8, g: u8, b: u8, a: u8) {
    if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
        let idx = ((y as u32) * w + x as u32) as usize * 4;
        // Alpha-blend if existing pixel is opaque
        buf[idx] = r;
        buf[idx + 1] = g;
        buf[idx + 2] = b;
        buf[idx + 3] = a;
    }
}

/// Draw a single character at (x, y) top-left corner, with a given scale factor.
fn draw_char(
    buf: &mut [u8],
    w: u32,
    h: u32,
    ch: char,
    x: i32,
    y: i32,
    scale: u32,
    r: u8,
    g: u8,
    b: u8,
) {
    if let Some(gly) = glyph(ch) {
        for row in 0..7u32 {
            let bits = gly[row as usize];
            for col in 0..5u32 {
                if bits & (1 << (4 - col)) != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            put_pixel(
                                buf,
                                w,
                                h,
                                x + (col * scale + sx) as i32,
                                y + (row * scale + sy) as i32,
                                r,
                                g,
                                b,
                                255,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Draw a string at (x, y) top-left. Returns width in pixels.
fn draw_string(
    buf: &mut [u8],
    w: u32,
    h: u32,
    s: &str,
    x: i32,
    y: i32,
    scale: u32,
    r: u8,
    g: u8,
    b: u8,
) -> i32 {
    let char_w = (5 * scale + scale) as i32; // 5 pixels + 1 pixel spacing, scaled
    let mut cx = x;
    for ch in s.chars() {
        draw_char(buf, w, h, ch, cx, y, scale, r, g, b);
        cx += char_w;
    }
    cx - x
}

/// Width of a string in pixels at a given scale.
fn string_width(s: &str, scale: u32) -> i32 {
    let n = s.chars().count() as i32;
    if n == 0 {
        return 0;
    }
    n * (5 * scale as i32 + scale as i32) - scale as i32
}

/// Draw a line using Bresenham's algorithm.
fn draw_line(
    buf: &mut [u8],
    w: u32,
    h: u32,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    r: u8,
    g: u8,
    b: u8,
) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut cx = x0;
    let mut cy = y0;

    loop {
        put_pixel(buf, w, h, cx, cy, r, g, b, 255);
        if cx == x1 && cy == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            cx += sx;
        }
        if e2 <= dx {
            err += dx;
            cy += sy;
        }
    }
}

/// Draw a thick line (by drawing parallel lines offset perpendicular).
fn draw_thick_line(
    buf: &mut [u8],
    w: u32,
    h: u32,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    thickness: i32,
    r: u8,
    g: u8,
    b: u8,
) {
    let half = thickness / 2;
    // Simple approach: draw line for each offset
    let dx = (x1 - x0) as f64;
    let dy = (y1 - y0) as f64;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.5 {
        put_pixel(buf, w, h, x0, y0, r, g, b, 255);
        return;
    }
    // Perpendicular unit vector
    let px = -dy / len;
    let py = dx / len;

    for t in -half..=half {
        let ox = (px * t as f64).round() as i32;
        let oy = (py * t as f64).round() as i32;
        draw_line(buf, w, h, x0 + ox, y0 + oy, x1 + ox, y1 + oy, r, g, b);
    }
}

/// Draw a circle outline.
fn draw_circle(
    buf: &mut [u8],
    w: u32,
    h: u32,
    cx: i32,
    cy: i32,
    radius: i32,
    r: u8,
    g: u8,
    b: u8,
) {
    // Midpoint circle algorithm
    let mut x = radius;
    let mut y = 0i32;
    let mut err = 1 - radius;

    while x >= y {
        put_pixel(buf, w, h, cx + x, cy + y, r, g, b, 255);
        put_pixel(buf, w, h, cx - x, cy + y, r, g, b, 255);
        put_pixel(buf, w, h, cx + x, cy - y, r, g, b, 255);
        put_pixel(buf, w, h, cx - x, cy - y, r, g, b, 255);
        put_pixel(buf, w, h, cx + y, cy + x, r, g, b, 255);
        put_pixel(buf, w, h, cx - y, cy + x, r, g, b, 255);
        put_pixel(buf, w, h, cx + y, cy - x, r, g, b, 255);
        put_pixel(buf, w, h, cx - y, cy - x, r, g, b, 255);
        y += 1;
        if err < 0 {
            err += 2 * y + 1;
        } else {
            x -= 1;
            err += 2 * (y - x) + 1;
        }
    }
}

/// Draw a filled circle.
fn fill_circle(
    buf: &mut [u8],
    w: u32,
    h: u32,
    cx: i32,
    cy: i32,
    radius: i32,
    r: u8,
    g: u8,
    b: u8,
) {
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= radius * radius {
                put_pixel(buf, w, h, cx + dx, cy + dy, r, g, b, 255);
            }
        }
    }
}

/// Draw sky cover circle: center station circle filled proportionally by oktas.
fn draw_sky_cover(
    buf: &mut [u8],
    w: u32,
    h: u32,
    cx: i32,
    cy: i32,
    radius: i32,
    oktas: u8,
) {
    let oktas = oktas.min(8);

    match oktas {
        0 => {
            // Clear: empty circle
            draw_circle(buf, w, h, cx, cy, radius, 0, 0, 0);
        }
        1 => {
            // 1/8: circle with small vertical line in center
            draw_circle(buf, w, h, cx, cy, radius, 0, 0, 0);
            draw_line(buf, w, h, cx, cy - radius / 3, cx, cy + radius / 3, 0, 0, 0);
        }
        2 => {
            // 2/8: quarter filled (upper-right)
            draw_circle(buf, w, h, cx, cy, radius, 0, 0, 0);
            for dy in -radius..=0 {
                for dx in 0..=radius {
                    if dx * dx + dy * dy <= radius * radius {
                        put_pixel(buf, w, h, cx + dx, cy + dy, 0, 0, 0, 255);
                    }
                }
            }
        }
        3 => {
            // 3/8: upper-right + small extra
            draw_circle(buf, w, h, cx, cy, radius, 0, 0, 0);
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dx * dx + dy * dy <= radius * radius {
                        // Fill right half top + a bit more
                        if (dy < 0 && dx >= 0) || (dy >= 0 && dx >= 0 && dy <= radius / 3) {
                            put_pixel(buf, w, h, cx + dx, cy + dy, 0, 0, 0, 255);
                        }
                    }
                }
            }
        }
        4 => {
            // Half: right side filled
            draw_circle(buf, w, h, cx, cy, radius, 0, 0, 0);
            for dy in -radius..=radius {
                for dx in 0..=radius {
                    if dx * dx + dy * dy <= radius * radius {
                        put_pixel(buf, w, h, cx + dx, cy + dy, 0, 0, 0, 255);
                    }
                }
            }
        }
        5 => {
            // 5/8: right side + upper-left quadrant
            draw_circle(buf, w, h, cx, cy, radius, 0, 0, 0);
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dx * dx + dy * dy <= radius * radius {
                        if dx >= 0 || dy < 0 {
                            put_pixel(buf, w, h, cx + dx, cy + dy, 0, 0, 0, 255);
                        }
                    }
                }
            }
        }
        6 => {
            // 6/8: mostly filled, lower-left open
            draw_circle(buf, w, h, cx, cy, radius, 0, 0, 0);
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dx * dx + dy * dy <= radius * radius {
                        if dx >= 0 || dy <= 0 || (dx < 0 && dy > 0 && -dx <= radius / 3) {
                            put_pixel(buf, w, h, cx + dx, cy + dy, 0, 0, 0, 255);
                        }
                    }
                }
            }
        }
        7 => {
            // 7/8: filled with small gap
            fill_circle(buf, w, h, cx, cy, radius, 0, 0, 0);
            // Erase a small notch on left
            for dy in 0..=radius / 3 {
                for dx in -radius..-radius / 2 {
                    if dx * dx + dy * dy <= radius * radius {
                        put_pixel(buf, w, h, cx + dx, cy + dy, 255, 255, 255, 255);
                    }
                }
            }
            draw_circle(buf, w, h, cx, cy, radius, 0, 0, 0);
        }
        8 => {
            // Overcast: completely filled
            fill_circle(buf, w, h, cx, cy, radius, 0, 0, 0);
        }
        _ => {
            draw_circle(buf, w, h, cx, cy, radius, 0, 0, 0);
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Wind barbs
// ═══════════════════════════════════════════════════════════

/// Draw a wind barb from the station circle outward.
///
/// Wind direction is where the wind comes FROM.
/// The barb shaft extends FROM the circle toward the wind source direction.
/// Barb ticks are on the left side of the shaft (NH convention).
///
/// Half barb = 5 kt, full barb = 10 kt, flag (triangle) = 50 kt.
fn draw_wind_barb(
    buf: &mut [u8],
    w: u32,
    h: u32,
    cx: i32,
    cy: i32,
    circle_radius: i32,
    speed_kt: f64,
    direction_deg: f64,
    barb_length: i32,
) {
    if speed_kt < 0.5 {
        // Calm: draw a larger concentric circle
        draw_circle(buf, w, h, cx, cy, circle_radius + 3, 0, 0, 0);
        return;
    }

    // Convert met direction to math angle
    // Met: 0=N (from north), 90=E (from east)
    // The shaft points TOWARD the direction the wind comes from
    let dir_rad = direction_deg.to_radians();
    // In pixel coordinates: x increases right, y increases down
    // North = up = -y direction
    // Met 0° (N) -> shaft points up -> angle = -PI/2 in standard math
    // We want: shaft_dx = -sin(dir), shaft_dy = cos(dir) ... wait
    // Actually: wind FROM north means shaft points upward (toward north)
    // shaft direction vector (in pixel coords, y-down):
    //   FROM north (0°): shaft points up => (0, -1)
    //   FROM east (90°): shaft points right => (1, 0) ... no, wait
    //   FROM east means wind comes from the east, shaft should point to the right (east)
    // Actually the shaft points INTO the wind (toward where wind comes from)
    //   FROM north: dx=0, dy=-1 (up)
    //   FROM east: dx=1, dy=0 ... no, east is to the right but from east means
    //   the wind source is to the east, so shaft extends toward east
    //   Actually wait - standard convention: shaft extends FROM station TOWARD source
    //   FROM north (0°): shaft up, dx=0, dy=-1
    //   FROM east (90°): shaft right, but east is right... but wind barbs typically
    //   show the shaft going to where wind comes from
    //   dx = sin(dir_deg), dy = -cos(dir_deg) in pixel coords (y-down)
    let shaft_dx = dir_rad.sin();
    let shaft_dy = -dir_rad.cos();

    // Shaft start: at edge of circle
    let sx = cx as f64 + shaft_dx * circle_radius as f64;
    let sy = cy as f64 + shaft_dy * circle_radius as f64;

    // Shaft end
    let ex = cx as f64 + shaft_dx * barb_length as f64;
    let ey = cy as f64 + shaft_dy * barb_length as f64;

    draw_thick_line(
        buf,
        w,
        h,
        sx.round() as i32,
        sy.round() as i32,
        ex.round() as i32,
        ey.round() as i32,
        2,
        0,
        0,
        0,
    );

    // Perpendicular direction for barb ticks (left side in NH convention)
    // Left of shaft direction (counterclockwise 90°)
    let perp_dx = -shaft_dy;
    let perp_dy = shaft_dx;

    let speed_rounded = (speed_kt / 5.0).round() as i32 * 5;
    let mut remaining = speed_rounded;
    let barb_tick_len = (barb_length as f64 * 0.35) as i32;
    let barb_spacing = (barb_length as f64 * 0.12).max(4.0);

    // Position along shaft from the tip (end) inward
    let mut pos = 0.0f64;

    // Draw 50-kt flags first
    while remaining >= 50 {
        let flag_start = pos;
        let flag_end = pos + barb_spacing * 1.5;

        // Triangle flag: three points
        let p1x = ex - shaft_dx * flag_start;
        let p1y = ey - shaft_dy * flag_start;
        let p2x = ex - shaft_dx * flag_end;
        let p2y = ey - shaft_dy * flag_end;
        let p3x = p1x + perp_dx * barb_tick_len as f64;
        let p3y = p1y + perp_dy * barb_tick_len as f64;

        // Fill the triangle
        fill_triangle(
            buf,
            w,
            h,
            p1x.round() as i32,
            p1y.round() as i32,
            p2x.round() as i32,
            p2y.round() as i32,
            p3x.round() as i32,
            p3y.round() as i32,
            0,
            0,
            0,
        );

        pos = flag_end + barb_spacing * 0.3;
        remaining -= 50;
    }

    // Draw full barbs (10 kt)
    while remaining >= 10 {
        let bx = ex - shaft_dx * pos;
        let by = ey - shaft_dy * pos;
        let tx = bx + perp_dx * barb_tick_len as f64;
        let ty = by + perp_dy * barb_tick_len as f64;

        draw_thick_line(
            buf,
            w,
            h,
            bx.round() as i32,
            by.round() as i32,
            tx.round() as i32,
            ty.round() as i32,
            2,
            0,
            0,
            0,
        );

        pos += barb_spacing;
        remaining -= 10;
    }

    // Draw half barb (5 kt)
    if remaining >= 5 {
        // If this is the only barb, offset it slightly from the tip
        if pos < 0.5 {
            pos += barb_spacing;
        }
        let bx = ex - shaft_dx * pos;
        let by = ey - shaft_dy * pos;
        let tx = bx + perp_dx * (barb_tick_len as f64 * 0.5);
        let ty = by + perp_dy * (barb_tick_len as f64 * 0.5);

        draw_thick_line(
            buf,
            w,
            h,
            bx.round() as i32,
            by.round() as i32,
            tx.round() as i32,
            ty.round() as i32,
            2,
            0,
            0,
            0,
        );
    }
}

/// Fill a triangle using scanline rasterization.
fn fill_triangle(
    buf: &mut [u8],
    w: u32,
    h: u32,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    r: u8,
    g: u8,
    b: u8,
) {
    let min_y = y0.min(y1).min(y2);
    let max_y = y0.max(y1).max(y2);
    let min_x = x0.min(x1).min(x2);
    let max_x = x0.max(x1).max(x2);

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            // Barycentric test
            let d = (y1 - y2) as f64 * (x0 - x2) as f64 + (x2 - x1) as f64 * (y0 - y2) as f64;
            if d.abs() < 0.001 {
                continue;
            }
            let a = ((y1 - y2) as f64 * (px - x2) as f64
                + (x2 - x1) as f64 * (py - y2) as f64)
                / d;
            let bb = ((y2 - y0) as f64 * (px - x2) as f64
                + (x0 - x2) as f64 * (py - y2) as f64)
                / d;
            let c = 1.0 - a - bb;
            if a >= -0.01 && bb >= -0.01 && c >= -0.01 {
                put_pixel(buf, w, h, px, py, r, g, b, 255);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Coordinate conversion
// ═══════════════════════════════════════════════════════════

/// Convert (lon, lat) to pixel (x, y).
#[inline]
fn geo_to_pixel(
    lon: f64,
    lat: f64,
    lon_min: f64,
    lon_max: f64,
    lat_min: f64,
    lat_max: f64,
    width: u32,
    height: u32,
) -> (i32, i32) {
    let x = ((lon - lon_min) / (lon_max - lon_min) * width as f64).round() as i32;
    // Latitude: higher lat = lower y (top of image)
    let y = ((lat_max - lat) / (lat_max - lat_min) * height as f64).round() as i32;
    (x, y)
}

// ═══════════════════════════════════════════════════════════
// Pressure formatting
// ═══════════════════════════════════════════════════════════

/// Format pressure for station model: last 3 digits of (pressure * 10).
/// e.g. 1013.2 -> "132", 998.7 -> "987", 1024.0 -> "240"
fn format_pressure(p: f64) -> String {
    let tenths = (p * 10.0).round() as i64;
    let last3 = ((tenths % 1000) + 1000) % 1000; // ensure positive
    format!("{:03}", last3)
}

/// Format temperature: round to integer.
fn format_temp(t: f64) -> String {
    let v = t.round() as i64;
    format!("{}", v)
}

// ═══════════════════════════════════════════════════════════
// Station thinning
// ═══════════════════════════════════════════════════════════

/// Thin stations so no two are closer than `radius` degrees.
/// Returns indices of stations to keep, prioritizing order (first come, first served).
fn thin_stations(stations: &[StationObs], radius: f64) -> Vec<usize> {
    let mut kept: Vec<usize> = Vec::new();
    let r2 = radius * radius;

    for i in 0..stations.len() {
        let mut too_close = false;
        for &j in &kept {
            let dlat = stations[i].lat - stations[j].lat;
            let dlon = stations[i].lon - stations[j].lon;
            if dlat * dlat + dlon * dlon < r2 {
                too_close = true;
                break;
            }
        }
        if !too_close {
            kept.push(i);
        }
    }

    kept
}

// ═══════════════════════════════════════════════════════════
// Main render function
// ═══════════════════════════════════════════════════════════

/// Render station model plots onto an RGBA pixel buffer.
///
/// # Arguments
/// * `stations` - Slice of station observations
/// * `config` - Plot configuration (bounds, size, thinning, etc.)
///
/// # Returns
/// RGBA pixel buffer of length `config.width * config.height * 4`.
pub fn render_station_plot(
    stations: &[StationObs],
    config: &StationPlotConfig,
) -> Vec<u8> {
    let w = config.width;
    let h = config.height;
    let mut buf = vec![0u8; (w * h * 4) as usize];

    // Fill background
    for i in 0..(w * h) as usize {
        buf[i * 4] = config.bg_color.0;
        buf[i * 4 + 1] = config.bg_color.1;
        buf[i * 4 + 2] = config.bg_color.2;
        buf[i * 4 + 3] = config.bg_color.3;
    }

    // Thin stations
    let kept = thin_stations(stations, config.thinning_radius);

    // Font scale: map font_size to a pixel multiplier for the 5x7 bitmap font
    let scale = ((config.font_size as u32).max(8) / 7).max(1);
    let char_h = 7 * scale;
    let circle_radius = (config.station_size / 8).max(3) as i32;
    let barb_length = (config.station_size as f64 * 0.6).round() as i32;

    for &idx in &kept {
        let obs = &stations[idx];

        // Skip if outside bounds
        if obs.lon < config.lon_min
            || obs.lon > config.lon_max
            || obs.lat < config.lat_min
            || obs.lat > config.lat_max
        {
            continue;
        }

        let (px, py) = geo_to_pixel(
            obs.lon,
            obs.lat,
            config.lon_min,
            config.lon_max,
            config.lat_min,
            config.lat_max,
            w,
            h,
        );

        // 1. Sky cover circle
        let oktas = obs.sky_cover.unwrap_or(0);
        draw_sky_cover(&mut buf, w, h, px, py, circle_radius, oktas);

        // 2. Wind barb
        if let (Some(spd), Some(dir)) = (obs.wind_speed, obs.wind_direction) {
            draw_wind_barb(
                &mut buf,
                w,
                h,
                px,
                py,
                circle_radius,
                spd,
                dir,
                barb_length,
            );
        }

        // 3. Temperature: upper-left of station circle
        if let Some(t) = obs.temperature {
            let text = format_temp(t);
            let tw = string_width(&text, scale);
            let tx = px - circle_radius as i32 - tw - (scale as i32 * 2);
            let ty = py - circle_radius as i32 - char_h as i32;
            draw_string(&mut buf, w, h, &text, tx, ty, scale, 200, 0, 0);
        }

        // 4. Dewpoint: lower-left of station circle
        if let Some(td) = obs.dewpoint {
            let text = format_temp(td);
            let tw = string_width(&text, scale);
            let tx = px - circle_radius as i32 - tw - (scale as i32 * 2);
            let ty = py + circle_radius as i32 + (scale as i32);
            draw_string(&mut buf, w, h, &text, tx, ty, scale, 0, 128, 0);
        }

        // 5. Pressure: upper-right of station circle
        if let Some(p) = obs.pressure {
            let text = format_pressure(p);
            let tx = px + circle_radius as i32 + (scale as i32 * 2);
            let ty = py - circle_radius as i32 - char_h as i32;
            draw_string(&mut buf, w, h, &text, tx, ty, scale, 0, 0, 0);
        }

        // 6. Pressure tendency: lower-right
        if let Some(pt) = obs.pressure_tendency {
            let text = format!("{:+.1}", pt);
            let tx = px + circle_radius as i32 + (scale as i32 * 2);
            let ty = py + circle_radius as i32 + (scale as i32);
            draw_string(&mut buf, w, h, &text, tx, ty, scale, 100, 100, 100);
        }
    }

    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_pressure() {
        assert_eq!(format_pressure(1013.2), "132");
        assert_eq!(format_pressure(998.7), "987");
        assert_eq!(format_pressure(1024.0), "240");
    }

    #[test]
    fn test_format_temp() {
        assert_eq!(format_temp(72.3), "72");
        assert_eq!(format_temp(-5.8), "-6");
    }

    #[test]
    fn test_thin_stations() {
        let stations = vec![
            StationObs {
                lat: 40.0,
                lon: -90.0,
                temperature: None,
                dewpoint: None,
                wind_speed: None,
                wind_direction: None,
                pressure: None,
                sky_cover: None,
                weather: None,
                visibility: None,
                pressure_tendency: None,
            },
            StationObs {
                lat: 40.01,
                lon: -90.01,
                temperature: None,
                dewpoint: None,
                wind_speed: None,
                wind_direction: None,
                pressure: None,
                sky_cover: None,
                weather: None,
                visibility: None,
                pressure_tendency: None,
            },
            StationObs {
                lat: 42.0,
                lon: -88.0,
                temperature: None,
                dewpoint: None,
                wind_speed: None,
                wind_direction: None,
                pressure: None,
                sky_cover: None,
                weather: None,
                visibility: None,
                pressure_tendency: None,
            },
        ];
        // With radius 0.5, first two are too close, so second is dropped
        let kept = thin_stations(&stations, 0.5);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0], 0);
        assert_eq!(kept[1], 2);
    }

    #[test]
    fn test_render_station_plot_basic() {
        let stations = vec![StationObs {
            lat: 40.0,
            lon: -90.0,
            temperature: Some(72.0),
            dewpoint: Some(55.0),
            wind_speed: Some(15.0),
            wind_direction: Some(270.0),
            pressure: Some(1013.2),
            sky_cover: Some(4),
            weather: None,
            visibility: Some(10.0),
            pressure_tendency: Some(-1.2),
        }];

        let config = StationPlotConfig {
            width: 200,
            height: 150,
            lon_min: -100.0,
            lon_max: -80.0,
            lat_min: 35.0,
            lat_max: 45.0,
            station_size: 60,
            font_size: 12,
            bg_color: (255, 255, 255, 255),
            thinning_radius: 0.0,
        };

        let pixels = render_station_plot(&stations, &config);
        assert_eq!(pixels.len(), 200 * 150 * 4);

        // Check that at least some non-background pixels were drawn
        let non_bg = pixels.chunks(4).filter(|p| p[0] != 255 || p[1] != 255 || p[2] != 255).count();
        assert!(non_bg > 0, "Should have drawn some station model pixels");
    }

    #[test]
    fn test_render_calm_wind() {
        let stations = vec![StationObs {
            lat: 40.0,
            lon: -90.0,
            temperature: Some(50.0),
            dewpoint: Some(45.0),
            wind_speed: Some(0.0),
            wind_direction: Some(0.0),
            pressure: None,
            sky_cover: Some(0),
            weather: None,
            visibility: None,
            pressure_tendency: None,
        }];

        let config = StationPlotConfig {
            width: 100,
            height: 100,
            lon_min: -95.0,
            lon_max: -85.0,
            lat_min: 35.0,
            lat_max: 45.0,
            station_size: 40,
            font_size: 10,
            bg_color: (255, 255, 255, 255),
            thinning_radius: 0.0,
        };

        let pixels = render_station_plot(&stations, &config);
        assert_eq!(pixels.len(), 100 * 100 * 4);
    }

    #[test]
    fn test_empty_stations() {
        let config = StationPlotConfig::default();
        let pixels = render_station_plot(&[], &config);
        assert_eq!(pixels.len(), (config.width * config.height * 4) as usize);
    }

    #[test]
    fn test_glyph_lookup() {
        assert!(glyph('0').is_some());
        assert!(glyph('9').is_some());
        assert!(glyph('-').is_some());
        assert!(glyph('.').is_some());
        assert!(glyph('A').is_some());
        assert!(glyph('~').is_none()); // not in our font
    }
}
