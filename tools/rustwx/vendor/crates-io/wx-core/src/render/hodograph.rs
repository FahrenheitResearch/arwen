//! Hodograph renderer.
//!
//! Produces a polar wind hodograph diagram showing the variation of wind
//! speed and direction with height. The trace is colored by altitude layer:
//! 0–3 km red, 3–6 km green, 6–9 km blue, 9+ km purple.

/// Configuration for hodograph rendering.
pub struct HodographConfig {
    /// Square image size in pixels.
    pub size: u32,
    /// Maximum wind speed shown (knots). Determines the outermost ring.
    pub max_speed: f64,
    /// Spacing between speed rings (knots).
    pub ring_interval: f64,
}

impl Default for HodographConfig {
    fn default() -> Self {
        Self {
            size: 400,
            max_speed: 60.0,
            ring_interval: 10.0,
        }
    }
}

/// Sounding data for hodograph plotting.
pub struct HodographData {
    /// Pressure levels in hPa.
    pub pressure: Vec<f64>,
    /// U-component of wind (knots, positive = from west).
    pub u_wind: Vec<f64>,
    /// V-component of wind (knots, positive = from south).
    pub v_wind: Vec<f64>,
}

/// Render a hodograph to RGBA pixels.
///
/// Returns a `Vec<u8>` of length `size * size * 4` (row-major RGBA).
pub fn render_hodograph(data: &HodographData, config: &HodographConfig) -> Vec<u8> {
    let s = config.size as usize;
    let mut pixels = vec![0u8; s * s * 4];

    // Fill with white background
    for i in 0..s * s {
        pixels[i * 4] = 255;
        pixels[i * 4 + 1] = 255;
        pixels[i * 4 + 2] = 255;
        pixels[i * 4 + 3] = 255;
    }

    let margin = 30usize;
    let plot_r = (s / 2 - margin) as f64; // radius of the plot area in pixels
    let cx = (s / 2) as i32;
    let cy = (s / 2) as i32;

    // Scale: knots to pixels
    let scale = plot_r / config.max_speed;

    // ── 1. Speed rings ──────────────────────────────────────────────

    let mut speed = config.ring_interval;
    while speed <= config.max_speed {
        let r = (speed * scale) as i32;
        draw_circle(&mut pixels, s, cx, cy, r, (200, 200, 200), 200);
        // Label the ring on the right side
        let label = format!("{}", speed as i32);
        draw_text_small(
            &mut pixels, s, s,
            cx + r + 2, cy - 3,
            &label, (120, 120, 120),
        );
        speed += config.ring_interval;
    }

    // ── 2. Crosshairs and cardinal directions ───────────────────────

    // Horizontal and vertical axes
    draw_line(&mut pixels, s, s, cx - plot_r as i32, cy, cx + plot_r as i32, cy, (180, 180, 180), 255);
    draw_line(&mut pixels, s, s, cx, cy - plot_r as i32, cx, cy + plot_r as i32, (180, 180, 180), 255);

    // Cardinal direction labels (N is up = negative v direction in screen coords)
    draw_text_small(&mut pixels, s, s, cx - 3, cy - plot_r as i32 - 12, "N", (60, 60, 60));
    draw_text_small(&mut pixels, s, s, cx + plot_r as i32 + 4, cy - 3, "E", (60, 60, 60));
    draw_text_small(&mut pixels, s, s, cx - 3, cy + plot_r as i32 + 4, "S", (60, 60, 60));
    draw_text_small(&mut pixels, s, s, cx - plot_r as i32 - 10, cy - 3, "W", (60, 60, 60));

    // ── 3. Compute approximate heights for coloring ─────────────────

    // Estimate heights using the hypsometric equation
    let heights = estimate_heights(&data.pressure);

    // ── 4. Wind trace colored by height ─────────────────────────────

    if data.u_wind.len() >= 2 && data.v_wind.len() >= 2 {
        let n = data.u_wind.len().min(data.v_wind.len()).min(data.pressure.len());

        for i in 0..n - 1 {
            let u0 = data.u_wind[i];
            let v0 = data.v_wind[i];
            let u1 = data.u_wind[i + 1];
            let v1 = data.v_wind[i + 1];

            // Convert: u positive = east, v positive = north
            // Screen: x right = east, y up = north (but y increases downward)
            let px0 = cx + (u0 * scale) as i32;
            let py0 = cy - (v0 * scale) as i32;
            let px1 = cx + (u1 * scale) as i32;
            let py1 = cy - (v1 * scale) as i32;

            let h_km = heights.get(i).copied().unwrap_or(0.0) / 1000.0;
            let color = height_color(h_km);

            draw_line_thick(&mut pixels, s, s, px0, py0, px1, py1, color, 255, 2);
        }

        // Draw dots at each level
        for i in 0..n {
            let u = data.u_wind[i];
            let v = data.v_wind[i];
            let px = cx + (u * scale) as i32;
            let py = cy - (v * scale) as i32;
            let h_km = heights.get(i).copied().unwrap_or(0.0) / 1000.0;
            let color = height_color(h_km);
            fill_circle(&mut pixels, s, px, py, 3, color, 255);
        }

        // ── 5. Storm motion marker: Bunkers right-mover ────────────

        if let Some((rm_u, rm_v)) = bunkers_right_mover(data, &heights) {
            let px = cx + (rm_u * scale) as i32;
            let py = cy - (rm_v * scale) as i32;
            // Draw an X marker
            draw_line(&mut pixels, s, s, px - 5, py - 5, px + 5, py + 5, (0, 0, 0), 255);
            draw_line(&mut pixels, s, s, px - 5, py + 5, px + 5, py - 5, (0, 0, 0), 255);
            // Draw a circle around it
            draw_circle(&mut pixels, s, px, py, 7, (0, 0, 0), 255);
        }
    }

    pixels
}

/// Map height (km AGL) to a trace color.
fn height_color(h_km: f64) -> (u8, u8, u8) {
    if h_km < 3.0 {
        (200, 30, 30)   // red: 0-3 km
    } else if h_km < 6.0 {
        (30, 160, 30)   // green: 3-6 km
    } else if h_km < 9.0 {
        (30, 30, 200)   // blue: 6-9 km
    } else {
        (140, 30, 180)  // purple: 9+ km
    }
}

/// Estimate heights (m AGL) from pressure levels using the hypsometric equation.
/// Assumes a standard atmosphere temperature profile for simplicity.
fn estimate_heights(pressures: &[f64]) -> Vec<f64> {
    if pressures.is_empty() {
        return vec![];
    }
    let mut heights = vec![0.0_f64; pressures.len()];
    // Use scale height approximation: dz = -(R*T)/(g) * d(ln p)
    let r = 287.0; // J/(kg·K)
    let g = 9.81;
    let t_avg = 270.0; // K, rough average for troposphere

    for i in 1..pressures.len() {
        let dp_ln = (pressures[i - 1] / pressures[i]).ln();
        heights[i] = heights[i - 1] + r * t_avg / g * dp_ln;
    }

    heights
}

/// Compute Bunkers right-mover storm motion.
/// Uses the 0–6 km mean wind plus a 7.5 m/s deviation perpendicular
/// to the 0–6 km shear vector.
fn bunkers_right_mover(data: &HodographData, heights: &[f64]) -> Option<(f64, f64)> {
    let n = data.u_wind.len().min(data.v_wind.len()).min(heights.len());
    if n < 2 {
        return None;
    }

    // Find indices for 0–6 km layer
    let mut u_sum = 0.0_f64;
    let mut v_sum = 0.0_f64;
    let mut count = 0.0_f64;
    let idx_0km = 0usize;
    let mut idx_6km = 0usize;
    let mut found_6km = false;

    for i in 0..n {
        if heights[i] <= 6000.0 {
            u_sum += data.u_wind[i];
            v_sum += data.v_wind[i];
            count += 1.0;
            idx_6km = i;
            found_6km = true;
        }
    }

    if count < 2.0 || !found_6km {
        return None;
    }

    // Mean wind in 0-6 km
    let u_mean = u_sum / count;
    let v_mean = v_sum / count;

    // 0-6 km shear vector
    let du = data.u_wind[idx_6km] - data.u_wind[idx_0km];
    let dv = data.v_wind[idx_6km] - data.v_wind[idx_0km];
    let shear_mag = (du * du + dv * dv).sqrt();

    if shear_mag < 0.1 {
        return None;
    }

    // 7.5 m/s deviation perpendicular to shear, converted to knots
    let dev = 7.5 * 1.94384; // m/s to knots
    // Right-mover: perpendicular to right of shear vector
    let perp_u = dv / shear_mag * dev;
    let perp_v = -du / shear_mag * dev;

    Some((u_mean + perp_u, v_mean + perp_v))
}

// ── Drawing primitives (same style as skewt.rs) ─────────────────────

fn set_pixel(pixels: &mut [u8], s: usize, x: i32, y: i32, color: (u8, u8, u8), alpha: u8) {
    if x < 0 || y < 0 || x >= s as i32 || y >= s as i32 {
        return;
    }
    let idx = (y as usize * s + x as usize) * 4;
    if idx + 3 >= pixels.len() {
        return;
    }
    if alpha == 255 {
        pixels[idx] = color.0;
        pixels[idx + 1] = color.1;
        pixels[idx + 2] = color.2;
        pixels[idx + 3] = 255;
    } else {
        let a = alpha as f64 / 255.0;
        let inv_a = 1.0 - a;
        pixels[idx] = (color.0 as f64 * a + pixels[idx] as f64 * inv_a) as u8;
        pixels[idx + 1] = (color.1 as f64 * a + pixels[idx + 1] as f64 * inv_a) as u8;
        pixels[idx + 2] = (color.2 as f64 * a + pixels[idx + 2] as f64 * inv_a) as u8;
        pixels[idx + 3] = 255;
    }
}

/// Bresenham line drawing.
fn draw_line(pixels: &mut [u8], w: usize, h: usize, x0: i32, y0: i32, x1: i32, y1: i32, color: (u8, u8, u8), alpha: u8) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;

    loop {
        set_pixel(pixels, w.min(h), x, y, color, alpha);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            if x == x1 { break; }
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            if y == y1 { break; }
            err += dx;
            y += sy;
        }
    }
}

fn draw_line_thick(
    pixels: &mut [u8], w: usize, h: usize,
    x0: i32, y0: i32, x1: i32, y1: i32,
    color: (u8, u8, u8), alpha: u8, thickness: i32,
) {
    draw_line(pixels, w, h, x0, y0, x1, y1, color, alpha);
    for t in 1..=thickness {
        let dx = (x1 - x0) as f64;
        let dy = (y1 - y0) as f64;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 0.001 {
            break;
        }
        let nx = (-dy / len * t as f64) as i32;
        let ny = (dx / len * t as f64) as i32;
        draw_line(pixels, w, h, x0 + nx, y0 + ny, x1 + nx, y1 + ny, color, alpha);
        draw_line(pixels, w, h, x0 - nx, y0 - ny, x1 - nx, y1 - ny, color, alpha);
    }
}

/// Draw a circle outline (midpoint algorithm).
fn draw_circle(pixels: &mut [u8], s: usize, cx: i32, cy: i32, r: i32, color: (u8, u8, u8), alpha: u8) {
    let mut x = r;
    let mut y = 0;
    let mut err = 1 - r;

    while x >= y {
        set_pixel(pixels, s, cx + x, cy + y, color, alpha);
        set_pixel(pixels, s, cx - x, cy + y, color, alpha);
        set_pixel(pixels, s, cx + x, cy - y, color, alpha);
        set_pixel(pixels, s, cx - x, cy - y, color, alpha);
        set_pixel(pixels, s, cx + y, cy + x, color, alpha);
        set_pixel(pixels, s, cx - y, cy + x, color, alpha);
        set_pixel(pixels, s, cx + y, cy - x, color, alpha);
        set_pixel(pixels, s, cx - y, cy - x, color, alpha);
        y += 1;
        if err < 0 {
            err += 2 * y + 1;
        } else {
            x -= 1;
            err += 2 * (y - x) + 1;
        }
    }
}

/// Fill a circle.
fn fill_circle(pixels: &mut [u8], s: usize, cx: i32, cy: i32, r: i32, color: (u8, u8, u8), alpha: u8) {
    for dy in -r..=r {
        let max_dx = ((r * r - dy * dy) as f64).sqrt() as i32;
        for dx in -max_dx..=max_dx {
            set_pixel(pixels, s, cx + dx, cy + dy, color, alpha);
        }
    }
}

/// Draw small text using a 5x7 bitmap font.
fn draw_text_small(pixels: &mut [u8], w: usize, h: usize, x: i32, y: i32, text: &str, color: (u8, u8, u8)) {
    let s = w.min(h);
    let mut cx = x;
    for ch in text.chars() {
        let glyph = get_glyph(ch);
        for row in 0..7 {
            for col in 0..5 {
                if glyph[row] & (1 << (4 - col)) != 0 {
                    set_pixel(pixels, s, cx + col as i32, y + row as i32, color, 255);
                }
            }
        }
        cx += 6;
    }
}

fn get_glyph(ch: char) -> [u8; 7] {
    match ch {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        '3' => [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        ' ' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'S' => [0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        _ => [0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = HodographConfig::default();
        assert_eq!(cfg.size, 400);
        assert_eq!(cfg.max_speed, 60.0);
    }

    #[test]
    fn test_render_basic_hodograph() {
        let data = HodographData {
            pressure: vec![1000.0, 850.0, 700.0, 500.0, 300.0, 200.0],
            u_wind: vec![5.0, 10.0, 20.0, 30.0, 40.0, 45.0],
            v_wind: vec![0.0, 5.0, 10.0, 15.0, 10.0, 5.0],
        };
        let config = HodographConfig::default();
        let pixels = render_hodograph(&data, &config);
        assert_eq!(pixels.len(), 400 * 400 * 4);

        // Verify some non-white pixels exist
        let non_white = pixels.chunks(4).any(|p| p[0] != 255 || p[1] != 255 || p[2] != 255);
        assert!(non_white);
    }

    #[test]
    fn test_estimate_heights() {
        let pressures = vec![1000.0, 850.0, 500.0];
        let heights = estimate_heights(&pressures);
        assert_eq!(heights.len(), 3);
        assert_eq!(heights[0], 0.0);
        assert!(heights[1] > 0.0);
        assert!(heights[2] > heights[1]);
        // 500 hPa should be roughly around 5-6 km
        assert!(heights[2] > 4000.0 && heights[2] < 7000.0,
            "500 hPa height should be ~5.5km, got {} m", heights[2]);
    }

    #[test]
    fn test_height_color() {
        assert_eq!(height_color(1.0), (200, 30, 30));   // red
        assert_eq!(height_color(4.0), (30, 160, 30));    // green
        assert_eq!(height_color(7.0), (30, 30, 200));    // blue
        assert_eq!(height_color(10.0), (140, 30, 180));   // purple
    }

    #[test]
    fn test_bunkers_no_data() {
        let data = HodographData {
            pressure: vec![],
            u_wind: vec![],
            v_wind: vec![],
        };
        let heights = estimate_heights(&data.pressure);
        assert!(bunkers_right_mover(&data, &heights).is_none());
    }
}
