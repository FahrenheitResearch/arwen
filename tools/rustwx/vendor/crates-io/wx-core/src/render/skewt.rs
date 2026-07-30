//! Skew-T Log-P diagram renderer.
//!
//! Produces a complete Skew-T Log-P thermodynamic diagram from a sounding
//! profile. Draws isotherms, isobars, dry adiabats, moist adiabats,
//! mixing ratio lines, temperature/dewpoint traces, wind barbs, and
//! CAPE/CIN shading.

/// Configuration for Skew-T rendering.
pub struct SkewTConfig {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Top pressure in hPa (lowest pressure plotted).
    pub p_min: f64,
    /// Bottom pressure in hPa (highest pressure plotted).
    pub p_max: f64,
    /// Left temperature bound in °C.
    pub t_min: f64,
    /// Right temperature bound in °C.
    pub t_max: f64,
    /// Skew angle in degrees.
    pub skew_angle: f64,
    /// Background color (R, G, B).
    pub bg_color: (u8, u8, u8),
}

impl Default for SkewTConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 800,
            p_min: 100.0,
            p_max: 1050.0,
            t_min: -40.0,
            t_max: 50.0,
            skew_angle: 45.0,
            bg_color: (255, 255, 255),
        }
    }
}

/// Sounding data for Skew-T plotting.
pub struct SkewTData {
    /// Pressure levels in hPa, from surface (high) to top (low).
    pub pressure: Vec<f64>,
    /// Temperature at each level in °C.
    pub temperature: Vec<f64>,
    /// Dewpoint at each level in °C.
    pub dewpoint: Vec<f64>,
    /// Optional wind speed at each level in knots.
    pub wind_speed: Option<Vec<f64>>,
    /// Optional wind direction at each level in degrees (meteorological).
    pub wind_dir: Option<Vec<f64>>,
}

/// Render a complete Skew-T Log-P diagram to RGBA pixels.
///
/// Returns a `Vec<u8>` of length `width * height * 4` (row-major RGBA).
pub fn render_skewt(data: &SkewTData, config: &SkewTConfig) -> Vec<u8> {
    let w = config.width as usize;
    let h = config.height as usize;
    let mut pixels = vec![0u8; w * h * 4];

    // Fill background
    fill_rect(&mut pixels, w, h, 0, 0, w, h, config.bg_color, 255);

    // Margins for the plot area (leave room for labels and wind barbs)
    let margin_left = 60usize;
    let margin_right = 80usize; // room for wind barbs
    let margin_top = 30usize;
    let margin_bottom = 40usize;
    let plot_w = w.saturating_sub(margin_left + margin_right);
    let plot_h = h.saturating_sub(margin_top + margin_bottom);

    if plot_w < 10 || plot_h < 10 {
        return pixels;
    }

    let ln_p_max = config.p_max.ln();
    let ln_p_min = config.p_min.ln();
    let ln_range = ln_p_max - ln_p_min;
    let skew_tan = (config.skew_angle * std::f64::consts::PI / 180.0).tan();
    let t_range = config.t_max - config.t_min;

    // Coordinate transforms: (T °C, P hPa) -> pixel (px, py)
    let p_to_py = |p: f64| -> f64 {
        let frac = (ln_p_max - p.ln()) / ln_range;
        margin_top as f64 + frac * plot_h as f64
    };

    let tp_to_px = |t: f64, p: f64| -> f64 {
        let py = p_to_py(p);
        let frac_from_bottom = (py - margin_top as f64) / plot_h as f64;
        let skew_offset = (1.0 - frac_from_bottom) * plot_w as f64 * skew_tan;
        let t_frac = (t - config.t_min) / t_range;
        margin_left as f64 + t_frac * plot_w as f64 + skew_offset
    };

    // ── 1. Background grid ──────────────────────────────────────────

    // Isobars: horizontal lines at standard pressure levels
    let isobar_levels: &[f64] = &[
        1050.0, 1000.0, 950.0, 900.0, 850.0, 800.0, 750.0, 700.0,
        650.0, 600.0, 550.0, 500.0, 450.0, 400.0, 350.0, 300.0,
        250.0, 200.0, 150.0, 100.0,
    ];
    for &p in isobar_levels {
        if p < config.p_min || p > config.p_max {
            continue;
        }
        let py = p_to_py(p) as i32;
        draw_hline(&mut pixels, w, h, margin_left as i32, (margin_left + plot_w) as i32, py, (200, 200, 200), 255);
        // Label on left
        let label = format!("{}", p as i32);
        draw_text_small(&mut pixels, w, h, 2, py - 3, &label, (100, 100, 100));
    }

    // Isotherms: skewed vertical lines every 10°C
    let mut t = -80.0_f64;
    while t <= 60.0 {
        let color = if t == 0.0 { (0, 0, 200) } else { (200, 200, 230) };
        // Draw from p_max to p_min
        let x0 = tp_to_px(t, config.p_max) as i32;
        let y0 = p_to_py(config.p_max) as i32;
        let x1 = tp_to_px(t, config.p_min) as i32;
        let y1 = p_to_py(config.p_min) as i32;
        draw_line(&mut pixels, w, h, x0, y0, x1, y1, color, 255);
        // Label at bottom if in range
        if x0 > margin_left as i32 && x0 < (margin_left + plot_w) as i32 {
            let label = format!("{}", t as i32);
            draw_text_small(&mut pixels, w, h, x0 - 6, (margin_top + plot_h + 5) as i32, &label, (80, 80, 80));
        }
        t += 10.0;
    }

    // Dry adiabats (lines of constant potential temperature)
    // θ = T * (1000/P)^0.286   =>   T = θ * (P/1000)^0.286
    let dry_thetas: Vec<f64> = (-40..=80).step_by(10).map(|v| (v + 273) as f64).collect();
    for theta in &dry_thetas {
        let mut prev: Option<(i32, i32)> = None;
        let mut p = config.p_max;
        while p >= config.p_min {
            let t_c = theta * (p / 1000.0_f64).powf(0.286) - 273.15;
            let px = tp_to_px(t_c, p) as i32;
            let py = p_to_py(p) as i32;
            if let Some((px0, py0)) = prev {
                if in_plot(px, py, margin_left, margin_top, plot_w, plot_h)
                    || in_plot(px0, py0, margin_left, margin_top, plot_w, plot_h)
                {
                    draw_line(&mut pixels, w, h, px0, py0, px, py, (220, 180, 180), 200);
                }
            }
            prev = Some((px, py));
            p -= 10.0;
        }
    }

    // Moist adiabats (pseudo-adiabatic lapse rates)
    let moist_thetas: Vec<f64> = (-20..=36).step_by(4).map(|v| v as f64).collect();
    for &tw_start in &moist_thetas {
        let mut prev: Option<(i32, i32)> = None;
        let mut t_c = tw_start;
        let mut p = config.p_max;
        while p >= config.p_min {
            let px = tp_to_px(t_c, p) as i32;
            let py = p_to_py(p) as i32;
            if let Some((px0, py0)) = prev {
                if in_plot(px, py, margin_left, margin_top, plot_w, plot_h)
                    || in_plot(px0, py0, margin_left, margin_top, plot_w, plot_h)
                {
                    draw_line(&mut pixels, w, h, px0, py0, px, py, (180, 220, 180), 180);
                }
            }
            prev = Some((px, py));
            // Approximate moist-adiabatic lapse rate
            let es = 6.112 * (17.67 * t_c / (t_c + 243.5)).exp();
            let rs = 0.622 * es / (p - es);
            let lv = 2.501e6;
            let cp = 1004.0;
            let rd = 287.0;
            let t_k = t_c + 273.15;
            let gamma_m = (rd * t_k / (cp * p * 100.0))
                * (1.0 + lv * rs / (rd * t_k))
                / (1.0 + lv * lv * rs / (cp * 461.5 * t_k * t_k));
            // dT for a dp step (going up, p decreasing)
            let dp = 10.0; // hPa step
            t_c -= gamma_m * dp * 100.0; // convert hPa to Pa
            p -= dp;
        }
    }

    // Mixing ratio lines
    let mix_ratios: &[f64] = &[0.4, 1.0, 2.0, 4.0, 7.0, 10.0, 16.0, 24.0];
    for &w_g in mix_ratios {
        let mut prev: Option<(i32, i32)> = None;
        let mut p = config.p_max;
        while p >= config.p_min {
            // T_d from mixing ratio: es = w * p / (0.622 + w)
            let w_kg = w_g / 1000.0;
            let es = w_kg * p / (0.622 + w_kg);
            let td = dewpoint_from_es(es);
            let px = tp_to_px(td, p) as i32;
            let py = p_to_py(p) as i32;
            if let Some((px0, py0)) = prev {
                if in_plot(px, py, margin_left, margin_top, plot_w, plot_h)
                    || in_plot(px0, py0, margin_left, margin_top, plot_w, plot_h)
                {
                    draw_line_dashed(&mut pixels, w as usize, h, px0, py0, px, py, (180, 200, 180), 150, 6);
                }
            }
            prev = Some((px, py));
            p -= 25.0;
        }
    }

    // ── 2. CAPE and CIN shading ─────────────────────────────────────

    // Compute a simple parcel trace (surface-based)
    if data.pressure.len() >= 2 {
        let parcel_trace = compute_parcel_trace(data);
        if !parcel_trace.is_empty() {
            // Find LFC and EL by comparing parcel to environment temperature
            for i in 0..parcel_trace.len().saturating_sub(1) {
                let (p0, tp0) = parcel_trace[i];
                let (p1, tp1) = parcel_trace[i + 1];
                let te0 = interp_sounding(&data.pressure, &data.temperature, p0);
                let te1 = interp_sounding(&data.pressure, &data.temperature, p1);

                if let (Some(te0), Some(te1)) = (te0, te1) {
                    let x_parcel_0 = tp_to_px(tp0, p0) as i32;
                    let y0 = p_to_py(p0) as i32;
                    let x_env_0 = tp_to_px(te0, p0) as i32;
                    let x_parcel_1 = tp_to_px(tp1, p1) as i32;
                    let y1 = p_to_py(p1) as i32;
                    let x_env_1 = tp_to_px(te1, p1) as i32;

                    if tp0 > te0 || tp1 > te1 {
                        // CAPE region: parcel warmer than environment
                        fill_between_lines(
                            &mut pixels, w, h,
                            x_parcel_0, x_parcel_1, x_env_0, x_env_1,
                            y0, y1,
                            (255, 150, 150), 80,
                        );
                    } else if tp0 < te0 || tp1 < te1 {
                        // CIN region: parcel cooler than environment
                        fill_between_lines(
                            &mut pixels, w, h,
                            x_parcel_0, x_parcel_1, x_env_0, x_env_1,
                            y0, y1,
                            (150, 150, 255), 60,
                        );
                    }
                }
            }
        }
    }

    // ── 3. Temperature trace (red) ──────────────────────────────────

    for i in 0..data.pressure.len().saturating_sub(1) {
        let p0 = data.pressure[i];
        let p1 = data.pressure[i + 1];
        let t0 = data.temperature[i];
        let t1 = data.temperature[i + 1];
        if p0 < config.p_min || p0 > config.p_max || p1 < config.p_min || p1 > config.p_max {
            continue;
        }
        let x0 = tp_to_px(t0, p0) as i32;
        let y0 = p_to_py(p0) as i32;
        let x1 = tp_to_px(t1, p1) as i32;
        let y1 = p_to_py(p1) as i32;
        draw_line_thick(&mut pixels, w, h, x0, y0, x1, y1, (220, 20, 20), 255, 2);
    }

    // ── 4. Dewpoint trace (green) ───────────────────────────────────

    for i in 0..data.pressure.len().saturating_sub(1) {
        let p0 = data.pressure[i];
        let p1 = data.pressure[i + 1];
        let td0 = data.dewpoint[i];
        let td1 = data.dewpoint[i + 1];
        if p0 < config.p_min || p0 > config.p_max || p1 < config.p_min || p1 > config.p_max {
            continue;
        }
        let x0 = tp_to_px(td0, p0) as i32;
        let y0 = p_to_py(p0) as i32;
        let x1 = tp_to_px(td1, p1) as i32;
        let y1 = p_to_py(p1) as i32;
        draw_line_thick(&mut pixels, w, h, x0, y0, x1, y1, (20, 160, 20), 255, 2);
    }

    // ── 5. Wind barbs on right side ─────────────────────────────────

    if let (Some(speeds), Some(dirs)) = (&data.wind_speed, &data.wind_dir) {
        let barb_x = (margin_left + plot_w + 20) as i32;
        // Thin the barbs so they don't overlap — pick levels roughly every 50 hPa
        let mut last_py = -100i32;
        for i in 0..data.pressure.len() {
            let p = data.pressure[i];
            if p < config.p_min || p > config.p_max {
                continue;
            }
            if i >= speeds.len() || i >= dirs.len() {
                break;
            }
            let py = p_to_py(p) as i32;
            if (py - last_py).abs() < 20 {
                continue;
            }
            last_py = py;
            draw_wind_barb(&mut pixels, w, h, barb_x, py, speeds[i], dirs[i], (40, 40, 40));
        }
    }

    // ── 6. Plot border ──────────────────────────────────────────────

    let bl = margin_left as i32;
    let bt = margin_top as i32;
    let br = (margin_left + plot_w) as i32;
    let bb = (margin_top + plot_h) as i32;
    draw_hline(&mut pixels, w, h, bl, br, bt, (0, 0, 0), 255);
    draw_hline(&mut pixels, w, h, bl, br, bb, (0, 0, 0), 255);
    draw_vline(&mut pixels, w, h, bl, bt, bb, (0, 0, 0), 255);
    draw_vline(&mut pixels, w, h, br, bt, bb, (0, 0, 0), 255);

    pixels
}

// ── Thermodynamic helpers ───────────────────────────────────────────

/// Compute dewpoint from saturation vapor pressure (hPa).
fn dewpoint_from_es(es: f64) -> f64 {
    if es <= 0.0 {
        return -273.15;
    }
    let ln_es = (es / 6.112).ln();
    243.5 * ln_es / (17.67 - ln_es)
}

/// Compute saturation vapor pressure from temperature (°C) in hPa.
fn saturation_vapor_pressure(t_c: f64) -> f64 {
    6.112 * (17.67 * t_c / (t_c + 243.5)).exp()
}

/// Compute saturation mixing ratio (kg/kg) from T (°C) and P (hPa).
#[allow(dead_code)]
fn sat_mixing_ratio(t_c: f64, p_hpa: f64) -> f64 {
    let es = saturation_vapor_pressure(t_c);
    0.622 * es / (p_hpa - es).max(0.001)
}

/// Linearly interpolate a sounding value at a given pressure level.
fn interp_sounding(pressures: &[f64], values: &[f64], p: f64) -> Option<f64> {
    if pressures.is_empty() || values.is_empty() || pressures.len() != values.len() {
        return None;
    }
    // Sounding pressures may be decreasing (surface to top)
    for i in 0..pressures.len() - 1 {
        let (p0, p1) = (pressures[i], pressures[i + 1]);
        if (p0 >= p && p >= p1) || (p1 >= p && p >= p0) {
            let frac = (p - p0) / (p1 - p0);
            return Some(values[i] + frac * (values[i + 1] - values[i]));
        }
    }
    None
}

/// Compute a simple surface-based parcel ascent trace.
/// Returns Vec<(pressure_hPa, temperature_°C)> along the parcel path.
fn compute_parcel_trace(data: &SkewTData) -> Vec<(f64, f64)> {
    if data.pressure.is_empty() || data.temperature.is_empty() || data.dewpoint.is_empty() {
        return vec![];
    }

    let mut trace = Vec::new();
    let sfc_t = data.temperature[0]; // °C
    let sfc_td = data.dewpoint[0]; // °C
    let sfc_p = data.pressure[0]; // hPa

    // Compute LCL using iterative method
    let sfc_t_k = sfc_t + 273.15;
    let sfc_td_k = sfc_td + 273.15;
    let lcl_t_k = 1.0 / (1.0 / (sfc_td_k - 56.0) + (sfc_t_k / sfc_td_k).ln() / 800.0) + 56.0;
    let lcl_p = sfc_p * (lcl_t_k / sfc_t_k).powf(1.0 / 0.286);
    let lcl_t_c = lcl_t_k - 273.15;

    // Dry adiabatic ascent from surface to LCL
    let theta = sfc_t_k * (1000.0 / sfc_p).powf(0.286);
    let mut p = sfc_p;
    while p > lcl_p && p > 50.0 {
        let t_c = theta * (p / 1000.0).powf(0.286) - 273.15;
        trace.push((p, t_c));
        p -= 5.0;
    }

    // Moist adiabatic ascent from LCL upward
    let mut t_c = lcl_t_c;
    p = lcl_p;
    while p > 100.0 {
        trace.push((p, t_c));
        let es = saturation_vapor_pressure(t_c);
        let rs = 0.622 * es / (p - es).max(0.001);
        let lv = 2.501e6;
        let cp = 1004.0;
        let rd = 287.0;
        let t_k = t_c + 273.15;
        let gamma_m = (rd * t_k / (cp * p * 100.0))
            * (1.0 + lv * rs / (rd * t_k))
            / (1.0 + lv * lv * rs / (cp * 461.5 * t_k * t_k));
        let dp = 5.0;
        t_c -= gamma_m * dp * 100.0;
        p -= dp;
    }

    trace
}

// ── Drawing primitives ──────────────────────────────────────────────

fn in_plot(px: i32, py: i32, ml: usize, mt: usize, pw: usize, ph: usize) -> bool {
    px >= ml as i32 && px <= (ml + pw) as i32 && py >= mt as i32 && py <= (mt + ph) as i32
}

fn set_pixel(pixels: &mut [u8], w: usize, h: usize, x: i32, y: i32, color: (u8, u8, u8), alpha: u8) {
    if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
        return;
    }
    let idx = (y as usize * w + x as usize) * 4;
    if idx + 3 >= pixels.len() {
        return;
    }
    if alpha == 255 {
        pixels[idx] = color.0;
        pixels[idx + 1] = color.1;
        pixels[idx + 2] = color.2;
        pixels[idx + 3] = 255;
    } else {
        // Alpha blend
        let a = alpha as f64 / 255.0;
        let inv_a = 1.0 - a;
        pixels[idx] = (color.0 as f64 * a + pixels[idx] as f64 * inv_a) as u8;
        pixels[idx + 1] = (color.1 as f64 * a + pixels[idx + 1] as f64 * inv_a) as u8;
        pixels[idx + 2] = (color.2 as f64 * a + pixels[idx + 2] as f64 * inv_a) as u8;
        pixels[idx + 3] = 255;
    }
}

fn fill_rect(
    pixels: &mut [u8], w: usize, h: usize,
    x0: usize, y0: usize, rw: usize, rh: usize,
    color: (u8, u8, u8), alpha: u8,
) {
    for y in y0..y0 + rh {
        for x in x0..x0 + rw {
            set_pixel(pixels, w, h, x as i32, y as i32, color, alpha);
        }
    }
}

fn draw_hline(pixels: &mut [u8], w: usize, h: usize, x0: i32, x1: i32, y: i32, color: (u8, u8, u8), alpha: u8) {
    let (start, end) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
    for x in start..=end {
        set_pixel(pixels, w, h, x, y, color, alpha);
    }
}

fn draw_vline(pixels: &mut [u8], w: usize, h: usize, x: i32, y0: i32, y1: i32, color: (u8, u8, u8), alpha: u8) {
    let (start, end) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };
    for y in start..=end {
        set_pixel(pixels, w, h, x, y, color, alpha);
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
        set_pixel(pixels, w, h, x, y, color, alpha);
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

/// Draw a thick line by drawing parallel lines offset perpendicular to the direction.
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

/// Dashed Bresenham line.
fn draw_line_dashed(
    pixels: &mut [u8], w: usize, h: usize,
    x0: i32, y0: i32, x1: i32, y1: i32,
    color: (u8, u8, u8), alpha: u8, dash_len: i32,
) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    let mut count = 0;

    loop {
        if (count / dash_len) % 2 == 0 {
            set_pixel(pixels, w, h, x, y, color, alpha);
        }
        count += 1;
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

/// Fill the area between two vertical line segments (for CAPE/CIN shading).
/// Segments go from y0 to y1 (top to bottom in pixels).
fn fill_between_lines(
    pixels: &mut [u8], w: usize, h: usize,
    x_left_top: i32, x_left_bot: i32,
    x_right_top: i32, x_right_bot: i32,
    y_top: i32, y_bot: i32,
    color: (u8, u8, u8), alpha: u8,
) {
    if y_top == y_bot {
        return;
    }
    let (yt, yb) = if y_top < y_bot { (y_top, y_bot) } else { (y_bot, y_top) };
    for y in yt..=yb {
        let frac = if yb == yt { 0.0 } else { (y - yt) as f64 / (yb - yt) as f64 };
        let xl = x_left_top as f64 + frac * (x_left_bot - x_left_top) as f64;
        let xr = x_right_top as f64 + frac * (x_right_bot - x_right_top) as f64;
        let (start, end) = if xl < xr { (xl as i32, xr as i32) } else { (xr as i32, xl as i32) };
        for x in start..=end {
            set_pixel(pixels, w, h, x, y, color, alpha);
        }
    }
}

/// Draw a wind barb at the given pixel position.
/// Speed in knots, direction in meteorological degrees (0=N, 90=E, etc).
fn draw_wind_barb(
    pixels: &mut [u8], w: usize, h: usize,
    cx: i32, cy: i32,
    speed: f64, direction: f64,
    color: (u8, u8, u8),
) {
    // Wind barb points INTO the wind (from the direction wind is coming from)
    let dir_rad = (direction + 180.0) * std::f64::consts::PI / 180.0;
    let staff_len = 25.0_f64;
    let barb_len = 12.0_f64;

    // Staff endpoint (tip)
    let tip_x = cx as f64 + staff_len * dir_rad.sin();
    let tip_y = cy as f64 - staff_len * dir_rad.cos();

    // Draw staff
    draw_line(pixels, w, h, cx, cy, tip_x as i32, tip_y as i32, color, 255);

    if speed < 2.5 {
        // Calm — draw a circle
        draw_circle(pixels, w, h, cx, cy, 3, color);
        return;
    }

    let mut remaining = speed;
    let mut pos = 0.0_f64; // fraction along the staff from tip

    // Flags (50 kt)
    while remaining >= 47.5 {
        let p0 = pos;
        let p1 = pos + 0.15;
        let fx0 = tip_x + p0 * (cx as f64 - tip_x);
        let fy0 = tip_y + p0 * (cy as f64 - tip_y);
        let fx1 = tip_x + p1 * (cx as f64 - tip_x);
        let fy1 = tip_y + p1 * (cy as f64 - tip_y);
        // Perpendicular offset
        let perp_x = -(cy as f64 - tip_y) / staff_len * barb_len;
        let perp_y = (cx as f64 - tip_x) / staff_len * barb_len;
        // Draw filled triangle
        draw_line(pixels, w, h, fx0 as i32, fy0 as i32, (fx0 + perp_x) as i32, (fy0 + perp_y) as i32, color, 255);
        draw_line(pixels, w, h, (fx0 + perp_x) as i32, (fy0 + perp_y) as i32, fx1 as i32, fy1 as i32, color, 255);
        draw_line(pixels, w, h, fx0 as i32, fy0 as i32, fx1 as i32, fy1 as i32, color, 255);
        // Fill the triangle
        fill_triangle(
            pixels, w, h,
            fx0 as i32, fy0 as i32,
            (fx0 + perp_x) as i32, (fy0 + perp_y) as i32,
            fx1 as i32, fy1 as i32,
            color, 255,
        );
        remaining -= 50.0;
        pos += 0.15;
    }

    // Full barbs (10 kt)
    while remaining >= 7.5 {
        let frac = pos;
        let bx = tip_x + frac * (cx as f64 - tip_x);
        let by = tip_y + frac * (cy as f64 - tip_y);
        let perp_x = -(cy as f64 - tip_y) / staff_len * barb_len;
        let perp_y = (cx as f64 - tip_x) / staff_len * barb_len;
        draw_line(pixels, w, h, bx as i32, by as i32, (bx + perp_x) as i32, (by + perp_y) as i32, color, 255);
        remaining -= 10.0;
        pos += 0.12;
    }

    // Half barb (5 kt)
    if remaining >= 2.5 {
        let frac = pos;
        let bx = tip_x + frac * (cx as f64 - tip_x);
        let by = tip_y + frac * (cy as f64 - tip_y);
        let perp_x = -(cy as f64 - tip_y) / staff_len * barb_len * 0.5;
        let perp_y = (cx as f64 - tip_x) / staff_len * barb_len * 0.5;
        draw_line(pixels, w, h, bx as i32, by as i32, (bx + perp_x) as i32, (by + perp_y) as i32, color, 255);
    }
}

/// Draw a circle outline (Bresenham midpoint algorithm).
fn draw_circle(pixels: &mut [u8], w: usize, h: usize, cx: i32, cy: i32, r: i32, color: (u8, u8, u8)) {
    let mut x = r;
    let mut y = 0;
    let mut err = 1 - r;

    while x >= y {
        set_pixel(pixels, w, h, cx + x, cy + y, color, 255);
        set_pixel(pixels, w, h, cx - x, cy + y, color, 255);
        set_pixel(pixels, w, h, cx + x, cy - y, color, 255);
        set_pixel(pixels, w, h, cx - x, cy - y, color, 255);
        set_pixel(pixels, w, h, cx + y, cy + x, color, 255);
        set_pixel(pixels, w, h, cx - y, cy + x, color, 255);
        set_pixel(pixels, w, h, cx + y, cy - x, color, 255);
        set_pixel(pixels, w, h, cx - y, cy - x, color, 255);
        y += 1;
        if err < 0 {
            err += 2 * y + 1;
        } else {
            x -= 1;
            err += 2 * (y - x) + 1;
        }
    }
}

/// Fill a triangle using scanline.
fn fill_triangle(
    pixels: &mut [u8], w: usize, h: usize,
    x0: i32, y0: i32, x1: i32, y1: i32, x2: i32, y2: i32,
    color: (u8, u8, u8), alpha: u8,
) {
    let min_y = y0.min(y1).min(y2).max(0);
    let max_y = y0.max(y1).max(y2).min(h as i32 - 1);
    for y in min_y..=max_y {
        let mut min_x = w as i32;
        let mut max_x = 0i32;
        // Check each edge
        for &(ax, ay, bx, by) in &[(x0, y0, x1, y1), (x1, y1, x2, y2), (x2, y2, x0, y0)] {
            if (ay <= y && by >= y) || (by <= y && ay >= y) {
                let dy = by - ay;
                if dy != 0 {
                    let x = ax + (y - ay) * (bx - ax) / dy;
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                }
            }
        }
        for x in min_x..=max_x {
            set_pixel(pixels, w, h, x, y, color, alpha);
        }
    }
}

// ── Tiny bitmap font (5x7 pixel glyphs) ────────────────────────────

/// Draw a small text string using a built-in 5x7 bitmap font.
fn draw_text_small(pixels: &mut [u8], w: usize, h: usize, x: i32, y: i32, text: &str, color: (u8, u8, u8)) {
    let mut cx = x;
    for ch in text.chars() {
        let glyph = get_glyph(ch);
        for row in 0..7 {
            for col in 0..5 {
                if glyph[row] & (1 << (4 - col)) != 0 {
                    set_pixel(pixels, w, h, cx + col as i32, y + row as i32, color, 255);
                }
            }
        }
        cx += 6; // 5 pixels wide + 1 pixel gap
    }
}

/// Get a 5x7 bitmap glyph for a character.
/// Each entry is a u8 where bits 4..0 represent columns left to right.
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
        'h' => [0b10000, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'a' => [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111],
        _ => [0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111], // box for unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = SkewTConfig::default();
        assert_eq!(cfg.width, 800);
        assert_eq!(cfg.height, 800);
    }

    #[test]
    fn test_render_minimal_sounding() {
        let data = SkewTData {
            pressure: vec![1000.0, 850.0, 700.0, 500.0, 300.0, 200.0],
            temperature: vec![25.0, 15.0, 5.0, -10.0, -30.0, -45.0],
            dewpoint: vec![20.0, 10.0, -5.0, -25.0, -42.0, -55.0],
            wind_speed: Some(vec![5.0, 15.0, 25.0, 40.0, 55.0, 70.0]),
            wind_dir: Some(vec![180.0, 200.0, 230.0, 250.0, 270.0, 280.0]),
        };
        let config = SkewTConfig::default();
        let pixels = render_skewt(&data, &config);
        assert_eq!(pixels.len(), 800 * 800 * 4);

        // Check that not all pixels are white (something was drawn)
        let non_white = pixels.chunks(4).any(|p| p[0] != 255 || p[1] != 255 || p[2] != 255);
        assert!(non_white, "Expected some non-white pixels in the rendered output");
    }

    #[test]
    fn test_render_no_wind() {
        let data = SkewTData {
            pressure: vec![1000.0, 500.0, 200.0],
            temperature: vec![25.0, -10.0, -45.0],
            dewpoint: vec![20.0, -25.0, -55.0],
            wind_speed: None,
            wind_dir: None,
        };
        let config = SkewTConfig::default();
        let pixels = render_skewt(&data, &config);
        assert_eq!(pixels.len(), 800 * 800 * 4);
    }

    #[test]
    fn test_dewpoint_from_es() {
        let td = dewpoint_from_es(6.112);
        assert!((td - 0.0).abs() < 0.5, "Dewpoint at 6.112 hPa should be near 0°C, got {}", td);
    }

    #[test]
    fn test_interp_sounding() {
        let p = vec![1000.0, 500.0];
        let t = vec![20.0, -10.0];
        let result = interp_sounding(&p, &t, 750.0).unwrap();
        assert!((result - 5.0).abs() < 0.01);
    }
}
