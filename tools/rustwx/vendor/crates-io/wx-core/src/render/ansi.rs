//! ANSI truecolor terminal renderer for RGBA pixel buffers.
//!
//! Two rendering modes:
//! - **half**: Simple upper-half-block (▀) renderer — fast, good quality
//! - **block**: img2ansi-style quadrant block renderer — 16 Unicode block characters,
//!   optimal fg/bg color selection per 2x2 cell, Floyd-Steinberg error diffusion,
//!   Redmean perceptual color distance. Significantly better output quality.

use std::fmt::Write;

// ── 16 quadrant block characters ────────────────────────────────────
// Index is a 4-bit bitmask: bit3=TL, bit2=TR, bit1=BL, bit0=BR
// 1 = foreground, 0 = background

const BLOCK_CHARS: [char; 16] = [
    ' ',        // 0b0000
    '\u{2598}', // 0b0001 ▘ upper-left
    '\u{259D}', // 0b0010 ▝ upper-right
    '\u{2580}', // 0b0011 ▀ upper-half
    '\u{2596}', // 0b0100 ▖ lower-left
    '\u{258C}', // 0b0101 ▌ left-half
    '\u{259E}', // 0b0110 ▞ diagonal
    '\u{259B}', // 0b0111 ▛ upper-left + upper-right + lower-left
    '\u{2597}', // 0b1000 ▗ lower-right
    '\u{259A}', // 0b1001 ▚ anti-diagonal
    '\u{2590}', // 0b1010 ▐ right-half
    '\u{259C}', // 0b1011 ▜ upper-left + upper-right + lower-right
    '\u{2584}', // 0b1100 ▄ lower-half
    '\u{2599}', // 0b1101 ▙ upper-left + lower-left + lower-right
    '\u{259F}', // 0b1110 ▟ upper-right + lower-left + lower-right
    '\u{2588}', // 0b1111 █ full block
];

// Quadrant membership for each pattern: [TL, TR, BL, BR] — true = foreground
const QUADRANTS: [[bool; 4]; 16] = [
    [false, false, false, false], // 0
    [true,  false, false, false], // 1
    [false, true,  false, false], // 2
    [true,  true,  false, false], // 3
    [false, false, true,  false], // 4
    [true,  false, true,  false], // 5
    [false, true,  true,  false], // 6
    [true,  true,  true,  false], // 7
    [false, false, false, true],  // 8
    [true,  false, false, true],  // 9
    [false, true,  false, true],  // 10
    [true,  true,  false, true],  // 11
    [false, false, true,  true],  // 12
    [true,  false, true,  true],  // 13
    [false, true,  true,  true],  // 14
    [true,  true,  true,  true],  // 15
];

// ── Public API ──────────────────────────────────────────────────────

/// Rendering mode for ANSI output.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AnsiMode {
    /// Simple upper-half-block renderer (fast)
    Half,
    /// img2ansi-style quadrant block renderer with error diffusion (higher quality)
    Block,
}

impl AnsiMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "block" | "quadrant" | "hq" | "img2ansi" => AnsiMode::Block,
            _ => AnsiMode::Half,
        }
    }
}

/// Convert an RGBA pixel buffer to ANSI truecolor art.
///
/// Dispatches to the selected rendering mode.
pub fn rgba_to_ansi(
    pixels: &[u8],
    width: u32,
    height: u32,
    target_width: u32,
) -> String {
    rgba_to_ansi_mode(pixels, width, height, target_width, AnsiMode::Half)
}

/// Convert an RGBA pixel buffer to ANSI truecolor art with mode selection.
pub fn rgba_to_ansi_mode(
    pixels: &[u8],
    width: u32,
    height: u32,
    target_width: u32,
    mode: AnsiMode,
) -> String {
    match mode {
        AnsiMode::Half => render_half_block(pixels, width, height, target_width),
        AnsiMode::Block => render_quadrant_block(pixels, width, height, target_width),
    }
}

// ── Half-block renderer (original) ──────────────────────────────────

fn render_half_block(
    pixels: &[u8],
    width: u32,
    height: u32,
    target_width: u32,
) -> String {
    let w = width as usize;
    let h = height as usize;
    let tw = target_width as usize;
    let scale = w as f64 / tw as f64;
    let th = ((h as f64 / scale) / 2.0).round() as usize;

    let mut out = String::with_capacity(tw * th * 44);
    let mut prev_fg: (u8, u8, u8) = (0, 0, 0);
    let mut prev_bg: (u8, u8, u8) = (0, 0, 0);
    let mut has_prev = false;

    for ty in 0..th {
        let top_y0 = ((ty * 2) as f64 * scale) as usize;
        let top_y1 = (((ty * 2 + 1) as f64 * scale) as usize).min(h);
        let bot_y0 = ((ty * 2 + 1) as f64 * scale) as usize;
        let bot_y1 = (((ty * 2 + 2) as f64 * scale) as usize).min(h);

        for tx in 0..tw {
            let x0 = (tx as f64 * scale) as usize;
            let x1 = (((tx + 1) as f64 * scale) as usize).min(w);

            let top = sample_box(pixels, w, x0, x1, top_y0, top_y1);
            let bot = if bot_y0 < h {
                sample_box(pixels, w, x0, x1, bot_y0, bot_y1)
            } else {
                [0; 3]
            };

            let top_black = top[0] == 0 && top[1] == 0 && top[2] == 0;
            let bot_black = bot[0] == 0 && bot[1] == 0 && bot[2] == 0;

            if top_black && bot_black {
                if has_prev {
                    out.push_str("\x1b[0m");
                    has_prev = false;
                }
                out.push(' ');
            } else {
                let fg = (top[0], top[1], top[2]);
                let bg = (bot[0], bot[1], bot[2]);
                if !has_prev || fg != prev_fg {
                    let _ = write!(out, "\x1b[38;2;{};{};{}m", fg.0, fg.1, fg.2);
                }
                if !has_prev || bg != prev_bg {
                    let _ = write!(out, "\x1b[48;2;{};{};{}m", bg.0, bg.1, bg.2);
                }
                out.push('\u{2580}');
                prev_fg = fg;
                prev_bg = bg;
                has_prev = true;
            }
        }
        out.push_str("\x1b[0m\n");
        has_prev = false;
    }
    out
}

// ── Quadrant block renderer (img2ansi-style) ────────────────────────

fn render_quadrant_block(
    pixels: &[u8],
    width: u32,
    height: u32,
    target_width: u32,
) -> String {
    let w = width as usize;
    let h = height as usize;
    let tw = target_width as usize;

    // Each terminal cell = 2x2 sub-pixels.
    // Match the half-block mode height: scale so that tw columns span the image width,
    // then derive sub-pixel grid height from the aspect ratio.
    let scale = w as f64 / tw as f64;
    let th = ((h as f64 / scale) / 2.0).round() as usize;
    let sub_w = tw * 2;
    let sub_h = th * 2;

    if th == 0 || tw == 0 {
        return String::new();
    }

    // Downsample to sub-pixel grid with alpha compositing against black
    let mut grid = vec![[0i16; 3]; sub_w * sub_h];
    let x_scale = w as f64 / sub_w as f64;
    let y_scale = h as f64 / sub_h as f64;

    for sy in 0..sub_h {
        let src_y0 = (sy as f64 * y_scale) as usize;
        let src_y1 = (((sy + 1) as f64 * y_scale) as usize).min(h);
        for sx in 0..sub_w {
            let src_x0 = (sx as f64 * x_scale) as usize;
            let src_x1 = (((sx + 1) as f64 * x_scale) as usize).min(w);
            let c = sample_box(pixels, w, src_x0, src_x1, src_y0, src_y1);
            grid[sy * sub_w + sx] = [c[0] as i16, c[1] as i16, c[2] as i16];
        }
    }

    // Render with Floyd-Steinberg error diffusion
    let mut out = String::with_capacity(tw * th * 44);
    let mut prev_fg: (u8, u8, u8) = (0, 0, 0);
    let mut prev_bg: (u8, u8, u8) = (0, 0, 0);
    let mut has_prev = false;

    for ty in 0..th {
        for tx in 0..tw {
            // Extract 2x2 sub-pixel block: [TL, TR, BL, BR]
            let tl_idx = (ty * 2) * sub_w + (tx * 2);
            let tr_idx = tl_idx + 1;
            let bl_idx = (ty * 2 + 1) * sub_w + (tx * 2);
            let br_idx = bl_idx + 1;

            let block = [
                grid[tl_idx],
                grid[tr_idx],
                grid[bl_idx],
                grid[br_idx],
            ];

            // Check if all black
            let all_black = block.iter().all(|c| c[0] <= 0 && c[1] <= 0 && c[2] <= 0);
            if all_black {
                if has_prev {
                    out.push_str("\x1b[0m");
                    has_prev = false;
                }
                out.push(' ');
                continue;
            }

            // Find best block character + fg/bg colors
            let (best_pattern, fg, bg) = find_best_block(&block);

            // Floyd-Steinberg: compute and diffuse error for each sub-pixel
            let indices = [tl_idx, tr_idx, bl_idx, br_idx];
            let positions = [
                (tx * 2, ty * 2),
                (tx * 2 + 1, ty * 2),
                (tx * 2, ty * 2 + 1),
                (tx * 2 + 1, ty * 2 + 1),
            ];

            for qi in 0..4 {
                let assigned = if QUADRANTS[best_pattern][qi] { fg } else { bg };
                let orig = block[qi];
                let err = [
                    orig[0] - assigned[0] as i16,
                    orig[1] - assigned[1] as i16,
                    orig[2] - assigned[2] as i16,
                ];

                let (sx, sy) = positions[qi];

                // Diffuse error to neighbors (Floyd-Steinberg weights)
                // Right: 7/16, Below-left: 3/16, Below: 5/16, Below-right: 1/16
                let neighbors: [(i32, i32, i16); 4] = [
                    (1, 0, 7),
                    (-1, 1, 3),
                    (0, 1, 5),
                    (1, 1, 1),
                ];

                for &(dx, dy, weight) in &neighbors {
                    let nx = sx as i32 + dx;
                    let ny = sy as i32 + dy;
                    if nx >= 0 && (nx as usize) < sub_w && ny >= 0 && (ny as usize) < sub_h {
                        let ni = ny as usize * sub_w + nx as usize;
                        // Only diffuse forward (to unprocessed cells)
                        if ni > indices[qi] {
                            for c in 0..3 {
                                grid[ni][c] = (grid[ni][c] + err[c] * weight / 16).clamp(0, 255);
                            }
                        }
                    }
                }
            }

            let ch = BLOCK_CHARS[best_pattern];

            // Emit ANSI codes
            let fg_tuple = (fg[0], fg[1], fg[2]);
            let bg_tuple = (bg[0], bg[1], bg[2]);

            if best_pattern == 15 {
                // Full block — only need foreground
                if !has_prev || fg_tuple != prev_fg {
                    let _ = write!(out, "\x1b[38;2;{};{};{}m", fg[0], fg[1], fg[2]);
                }
                if has_prev && prev_bg != (0, 0, 0) {
                    out.push_str("\x1b[49m");
                }
            } else if best_pattern == 0 {
                // Space — only need background
                if !has_prev || bg_tuple != prev_bg {
                    let _ = write!(out, "\x1b[48;2;{};{};{}m", bg[0], bg[1], bg[2]);
                }
            } else {
                if !has_prev || fg_tuple != prev_fg {
                    let _ = write!(out, "\x1b[38;2;{};{};{}m", fg[0], fg[1], fg[2]);
                }
                if !has_prev || bg_tuple != prev_bg {
                    let _ = write!(out, "\x1b[48;2;{};{};{}m", bg[0], bg[1], bg[2]);
                }
            }

            out.push(ch);
            prev_fg = fg_tuple;
            prev_bg = bg_tuple;
            has_prev = true;
        }
        out.push_str("\x1b[0m\n");
        has_prev = false;
    }
    out
}

/// Find the best block pattern + fg/bg colors for a 2x2 sub-pixel block.
/// Returns (pattern_index, fg_rgb, bg_rgb).
fn find_best_block(block: &[[i16; 3]; 4]) -> (usize, [u8; 3], [u8; 3]) {
    let mut best_err = f64::MAX;
    let mut best_pattern = 3; // default: upper half
    let mut best_fg = [0u8; 3];
    let mut best_bg = [0u8; 3];

    for pat in 0..16 {
        let q = &QUADRANTS[pat];

        // Average foreground and background colors
        let mut fg_sum = [0i32; 3];
        let mut bg_sum = [0i32; 3];
        let mut fg_count = 0i32;
        let mut bg_count = 0i32;

        for i in 0..4 {
            if q[i] {
                for c in 0..3 { fg_sum[c] += block[i][c] as i32; }
                fg_count += 1;
            } else {
                for c in 0..3 { bg_sum[c] += block[i][c] as i32; }
                bg_count += 1;
            }
        }

        let fg = if fg_count > 0 {
            [
                (fg_sum[0] / fg_count).clamp(0, 255) as u8,
                (fg_sum[1] / fg_count).clamp(0, 255) as u8,
                (fg_sum[2] / fg_count).clamp(0, 255) as u8,
            ]
        } else {
            [0, 0, 0]
        };

        let bg = if bg_count > 0 {
            [
                (bg_sum[0] / bg_count).clamp(0, 255) as u8,
                (bg_sum[1] / bg_count).clamp(0, 255) as u8,
                (bg_sum[2] / bg_count).clamp(0, 255) as u8,
            ]
        } else {
            [0, 0, 0]
        };

        // Compute total Redmean error across all 4 sub-pixels
        let mut total_err = 0.0;
        for i in 0..4 {
            let assigned = if q[i] { fg } else { bg };
            let pixel = [
                block[i][0].clamp(0, 255) as u8,
                block[i][1].clamp(0, 255) as u8,
                block[i][2].clamp(0, 255) as u8,
            ];
            total_err += redmean_distance(pixel, assigned);
        }

        if total_err < best_err {
            best_err = total_err;
            best_pattern = pat;
            best_fg = fg;
            best_bg = bg;
        }
    }

    (best_pattern, best_fg, best_bg)
}

/// Redmean color distance — fast perceptual color difference.
///
/// Weights red and blue channels based on average redness, with green
/// always weighted 4x. Approximates human color perception much faster than CIE LAB.
#[inline]
fn redmean_distance(c1: [u8; 3], c2: [u8; 3]) -> f64 {
    let rmean = (c1[0] as f64 + c2[0] as f64) / 2.0;
    let dr = c1[0] as f64 - c2[0] as f64;
    let dg = c1[1] as f64 - c2[1] as f64;
    let db = c1[2] as f64 - c2[2] as f64;
    ((2.0 + rmean / 256.0) * dr * dr + 4.0 * dg * dg + (2.0 + (255.0 - rmean) / 256.0) * db * db)
        .sqrt()
}

// ── Shared utilities ────────────────────────────────────────────────

/// Box-average sample a rectangular region of the RGBA buffer.
/// Returns composited-against-black RGB as [r, g, b].
#[inline]
fn sample_box(
    pixels: &[u8],
    stride: usize,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
) -> [u8; 3] {
    if x0 >= x1 || y0 >= y1 {
        return [0; 3];
    }

    let mut sr: u32 = 0;
    let mut sg: u32 = 0;
    let mut sb: u32 = 0;
    let mut sa: u32 = 0;
    let mut count: u32 = 0;

    for y in y0..y1 {
        for x in x0..x1 {
            let idx = (y * stride + x) * 4;
            sr += pixels[idx] as u32;
            sg += pixels[idx + 1] as u32;
            sb += pixels[idx + 2] as u32;
            sa += pixels[idx + 3] as u32;
            count += 1;
        }
    }

    if count == 0 {
        return [0; 3];
    }

    let r = sr / count;
    let g = sg / count;
    let b = sb / count;
    let a = sa / count;

    // Composite against black
    [
        ((r * a) / 255) as u8,
        ((g * a) / 255) as u8,
        ((b * a) / 255) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_half_block_basic() {
        let mut pixels = vec![0u8; 4 * 4 * 4];
        for i in 0..16 {
            pixels[i * 4] = 255;
            pixels[i * 4 + 3] = 255;
        }
        let result = rgba_to_ansi(&pixels, 4, 4, 2);
        assert!(result.contains("\u{2580}"));
        assert!(result.contains("38;2;255;0;0"));
    }

    #[test]
    fn test_transparent_renders_space() {
        let pixels = vec![0u8; 2 * 2 * 4];
        let result = rgba_to_ansi(&pixels, 2, 2, 2);
        assert!(!result.contains("\u{2580}"));
    }

    #[test]
    fn test_quadrant_block_basic() {
        // 4x4 image: top-left red, rest blue — should pick a quadrant char
        let mut pixels = vec![0u8; 4 * 4 * 4];
        for y in 0..4 {
            for x in 0..4 {
                let i = (y * 4 + x) * 4;
                if x < 2 && y < 2 {
                    pixels[i] = 255; // red
                } else {
                    pixels[i + 2] = 255; // blue
                }
                pixels[i + 3] = 255; // opaque
            }
        }
        let result = rgba_to_ansi_mode(&pixels, 4, 4, 2, AnsiMode::Block);
        // Should contain some block character
        assert!(result.len() > 10);
    }

    #[test]
    fn test_redmean_identical() {
        assert_eq!(redmean_distance([128, 128, 128], [128, 128, 128]), 0.0);
    }

    #[test]
    fn test_redmean_different() {
        let d = redmean_distance([255, 0, 0], [0, 255, 0]);
        assert!(d > 0.0);
    }
}
