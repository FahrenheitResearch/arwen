//! Standard meteorological colormaps.
//!
//! Each colormap is a slice of (value_fraction, r, g, b) control points where
//! value_fraction is in [0.0, 1.0]. Colors are linearly interpolated between
//! control points for smooth gradients.

/// A single control point: (normalized_position, r, g, b).
/// Position is in [0.0, 1.0] mapping linearly from vmin to vmax.
pub type ColorStop = (f64, u8, u8, u8);

/// Linearly interpolate a color from a colormap at a given normalized value t in [0, 1].
/// Returns (r, g, b). Values outside [0, 1] are clamped.
pub fn interpolate_color(colormap: &[ColorStop], value: f64) -> (u8, u8, u8) {
    if colormap.is_empty() {
        return (0, 0, 0);
    }
    if colormap.len() == 1 {
        return (colormap[0].1, colormap[0].2, colormap[0].3);
    }

    let t = value.clamp(0.0, 1.0);

    // Find the bracketing control points
    if t <= colormap[0].0 {
        return (colormap[0].1, colormap[0].2, colormap[0].3);
    }
    if t >= colormap[colormap.len() - 1].0 {
        let last = &colormap[colormap.len() - 1];
        return (last.1, last.2, last.3);
    }

    for i in 0..colormap.len() - 1 {
        let (t0, r0, g0, b0) = colormap[i];
        let (t1, r1, g1, b1) = colormap[i + 1];
        if t >= t0 && t <= t1 {
            let frac = if (t1 - t0).abs() < 1e-12 {
                0.0
            } else {
                (t - t0) / (t1 - t0)
            };
            let r = (r0 as f64 + (r1 as f64 - r0 as f64) * frac) as u8;
            let g = (g0 as f64 + (g1 as f64 - g0 as f64) * frac) as u8;
            let b = (b0 as f64 + (b1 as f64 - b0 as f64) * frac) as u8;
            return (r, g, b);
        }
    }

    let last = &colormap[colormap.len() - 1];
    (last.1, last.2, last.3)
}

// ============================================================
// Temperature colormap: cool blues -> white -> warm reds
// Designed for -40C to +50C / -40F to 120F
// ============================================================
pub static TEMPERATURE: &[ColorStop] = &[
    (0.000, 0x2b, 0x5d, 0x7e), // deep cold blue
    (0.111, 0x75, 0xa8, 0xb0),
    (0.222, 0xae, 0xe3, 0xdc),
    (0.278, 0xa0, 0xb8, 0xd6),
    (0.333, 0x96, 0x8b, 0xc5),
    (0.389, 0x82, 0x43, 0xb2), // purple for sub-zero
    (0.417, 0xa3, 0x43, 0xb3),
    (0.444, 0xf7, 0xf7, 0xff), // near-white at freezing
    (0.472, 0xa0, 0xb8, 0xd6),
    (0.500, 0x0f, 0x55, 0x75), // cool teal
    (0.556, 0x6d, 0x8c, 0x77),
    (0.611, 0xf8, 0xee, 0xa2), // warm yellow
    (0.667, 0xaa, 0x71, 0x4d), // warm brown
    (0.722, 0x5f, 0x00, 0x00), // dark red
    (0.778, 0x85, 0x2c, 0x40),
    (0.833, 0xb2, 0x8f, 0x85),
    (0.889, 0xe7, 0xe0, 0xda),
    (0.944, 0x95, 0x93, 0x91),
    (1.000, 0x45, 0x48, 0x44), // extreme heat gray
];

// ============================================================
// Precipitation colormap: white -> gray -> green -> blue -> purple -> red -> brown
// Designed for 0 to 15 inches
// ============================================================
pub static PRECIPITATION: &[ColorStop] = &[
    (0.000, 0xff, 0xff, 0xff), // white (no precip)
    (0.005, 0xdc, 0xdc, 0xdc),
    (0.020, 0xbe, 0xbe, 0xbe),
    (0.040, 0x9e, 0x9e, 0x9e),
    (0.060, 0x81, 0x81, 0x81), // gray trace
    (0.067, 0xb8, 0xf0, 0xc1), // light green
    (0.133, 0x15, 0x64, 0x71), // dark teal
    (0.200, 0x16, 0x4f, 0xba), // blue
    (0.333, 0xd8, 0xed, 0xf5), // light blue
    (0.400, 0xcf, 0xbd, 0xdd), // lavender
    (0.533, 0xa1, 0x34, 0xb1), // purple
    (0.600, 0xa4, 0x3c, 0x32), // red
    (0.733, 0xdd, 0x9c, 0x98), // pink
    (0.800, 0xf6, 0xf0, 0xa3), // yellow
    (0.900, 0x7e, 0x4b, 0x26), // brown
    (1.000, 0x54, 0x2f, 0x17), // dark brown
];

// ============================================================
// Wind speed colormap: white -> blue -> purple -> red -> yellow -> brown
// Designed for 0-175 kt
// ============================================================
pub static WIND: &[ColorStop] = &[
    (0.000, 0xff, 0xff, 0xff), // calm white
    (0.083, 0x87, 0xce, 0xfa), // light sky blue
    (0.167, 0x6a, 0x5a, 0xcd), // slate blue
    (0.250, 0xe6, 0x96, 0xdc), // orchid
    (0.333, 0xc8, 0x5a, 0xbe), // medium orchid
    (0.417, 0xa0, 0x14, 0x96), // dark magenta
    (0.500, 0xc8, 0x00, 0x28), // crimson
    (0.583, 0xdc, 0x28, 0x3c), // red
    (0.667, 0xf0, 0x50, 0x50), // coral
    (0.750, 0xfa, 0xf0, 0x64), // khaki
    (0.833, 0xdc, 0xbe, 0x46), // dark khaki
    (0.917, 0xbe, 0x8c, 0x28), // dark goldenrod
    (1.000, 0xa0, 0x5a, 0x0a), // saddle brown
];

// ============================================================
// Reflectivity colormap: professional NWS-style color table
// Range: -5 to 75 dBZ (80 dBZ span)
// ============================================================
pub static REFLECTIVITY: &[ColorStop] = &[
    (0.000, 0x00, 0x00, 0x00), // -5 dBZ: no echo (transparent)
    (0.063, 0x00, 0x00, 0x00), // 0 dBZ: still transparent
    (0.125, 0xD8, 0xE2, 0xF3), // 5 dBZ: pale blue-gray
    (0.188, 0xD8, 0xE2, 0xF3), // 10 dBZ: (216,226,243)
    (0.250, 0x87, 0xA9, 0xCA), // 15 dBZ: muted blue
    (0.313, 0x3B, 0x6D, 0xC1), // 20 dBZ: (59,109,193)
    (0.375, 0x0F, 0x50, 0x5F), // 25 dBZ: (15,80,95) dark teal
    (0.438, 0x76, 0x9B, 0x7C), // 30 dBZ: (118,155,124) sage green
    (0.500, 0xFF, 0xF3, 0x71), // 35 dBZ: (255,243,113) yellow
    (0.563, 0xED, 0xB3, 0x44), // 40 dBZ: (237,179,68) gold
    (0.625, 0xDB, 0x74, 0x17), // 45 dBZ: (219,116,23) orange
    (0.688, 0xCC, 0x00, 0x00), // 50 dBZ: (204,0,0) red
    (0.750, 0x76, 0x03, 0x0A), // 55 dBZ: (118,3,10) dark red
    (0.781, 0xA0, 0x37, 0xAF), // 57.5 dBZ: (160,55,175) purple
    (0.875, 0x82, 0x82, 0x82), // 65 dBZ: (130,130,130) gray
    (0.938, 0xE6, 0xE6, 0xE6), // 70 dBZ: (230,230,230) light gray
    (1.000, 0xFF, 0xFF, 0xFF), // 75 dBZ: white
];

// ============================================================
// CAPE colormap: gray -> teal -> yellow -> orange -> red -> purple -> pink -> rose
// Designed for 0-8000 J/kg
// ============================================================
pub static CAPE: &[ColorStop] = &[
    (0.000, 0xff, 0xff, 0xff), // white
    (0.071, 0x69, 0x69, 0x69), // gray
    (0.143, 0x37, 0x53, 0x6a), // steel blue
    (0.214, 0xa7, 0xc8, 0xce), // powder blue
    (0.286, 0xe9, 0xdd, 0x96), // khaki
    (0.357, 0xe1, 0x6f, 0x02), // dark orange
    (0.429, 0xdc, 0x41, 0x10), // red-orange
    (0.500, 0x8b, 0x09, 0x50), // dark magenta
    (0.571, 0x73, 0x08, 0x8a), // dark violet
    (0.643, 0xda, 0x99, 0xe7), // plum
    (0.714, 0xe9, 0xbe, 0xc3), // misty rose
    (0.786, 0xb2, 0x44, 0x5a), // palevioletred
    (0.857, 0x89, 0x3d, 0x48), // dark rose
    (1.000, 0xbc, 0x91, 0x95), // rosy brown
];

// ============================================================
// Relative humidity colormap: brown (dry) -> green -> blue (moist)
// Designed for 0-100%
// ============================================================
pub static RELATIVE_HUMIDITY: &[ColorStop] = &[
    (0.000, 0xa5, 0x73, 0x4d), // brown (dry)
    (0.100, 0x38, 0x2f, 0x28), // dark brown
    (0.200, 0x6e, 0x65, 0x59), // dim gray
    (0.300, 0xa5, 0x9b, 0x8e), // gray
    (0.400, 0xdd, 0xd1, 0xc3), // light gray
    (0.450, 0xc8, 0xd7, 0xc0), // pale green
    (0.700, 0x00, 0x4a, 0x2f), // dark green
    (0.900, 0x00, 0x41, 0x23), // darker green
    (1.000, 0x28, 0x58, 0x8c), // steel blue (saturated)
];

// ============================================================
// Vorticity / generic diverging colormap: gray -> white -> yellow -> red -> purple -> blue -> cyan
// ============================================================
pub static VORTICITY: &[ColorStop] = &[
    (0.000, 0x32, 0x32, 0x32), // dark gray (negative)
    (0.100, 0x70, 0x70, 0x70),
    (0.200, 0xa1, 0xa1, 0xa1),
    (0.300, 0xd6, 0xd6, 0xd6),
    (0.400, 0xff, 0xff, 0xff), // white (zero)
    (0.450, 0xfd, 0xd2, 0x44), // yellow
    (0.500, 0xfe, 0xa0, 0x00), // orange
    (0.550, 0xf1, 0x67, 0x02), // dark orange
    (0.600, 0xda, 0x24, 0x22), // red
    (0.650, 0xab, 0x02, 0x9b), // magenta
    (0.700, 0x78, 0x00, 0x8f), // purple
    (0.750, 0x44, 0x00, 0x8b), // dark purple
    (0.800, 0x00, 0x01, 0x60), // navy
    (0.850, 0x24, 0x44, 0x88), // steel blue
    (0.900, 0x4f, 0x85, 0xb2), // cadet blue
    (0.950, 0x73, 0xca, 0xdb), // medium turquoise
    (1.000, 0x91, 0xff, 0xfd), // cyan
];

// ============================================================
// Dewpoint colormap: browns (dry) -> greens -> blues/purple (moist)
// Designed for dewpoint temperatures, shifted from temperature scale
// ============================================================
pub static DEWPOINT: &[ColorStop] = &[
    (0.000, 0x8B, 0x45, 0x13), // saddle brown (very dry)
    (0.100, 0xA0, 0x6A, 0x3A), // brown
    (0.200, 0xC4, 0x9E, 0x6C), // tan
    (0.300, 0xD2, 0xC4, 0x8A), // pale goldenrod
    (0.400, 0x8F, 0xBC, 0x5E), // yellow-green
    (0.500, 0x4C, 0xAF, 0x50), // green
    (0.600, 0x2E, 0x7D, 0x32), // dark green
    (0.700, 0x00, 0x88, 0x88), // teal
    (0.800, 0x1E, 0x90, 0xFF), // dodger blue
    (0.900, 0x4B, 0x00, 0x82), // indigo
    (1.000, 0x80, 0x00, 0x80), // purple (very moist)
];

// ============================================================
// Pressure colormap: diverging blue-white-red centered on 1013 mb
// ============================================================
pub static PRESSURE: &[ColorStop] = &[
    (0.000, 0x08, 0x30, 0x6B), // deep blue (low pressure)
    (0.125, 0x21, 0x66, 0xAC),
    (0.250, 0x4B, 0x96, 0xD0),
    (0.375, 0x92, 0xC5, 0xDE),
    (0.450, 0xD1, 0xE5, 0xF0),
    (0.500, 0xF7, 0xF7, 0xF7), // white (1013 center)
    (0.550, 0xF4, 0xD1, 0xC0),
    (0.625, 0xDA, 0x8A, 0x67),
    (0.750, 0xC4, 0x4E, 0x34),
    (0.875, 0xA1, 0x25, 0x12),
    (1.000, 0x67, 0x00, 0x1F), // deep red (high pressure)
];

// ============================================================
// Snow accumulation colormap: white -> light blue -> dark blue/purple
// Designed for 0-36+ inches
// ============================================================
pub static SNOW: &[ColorStop] = &[
    (0.000, 0xFF, 0xFF, 0xFF), // white (trace)
    (0.100, 0xE0, 0xF0, 0xFF), // very light blue
    (0.200, 0xBE, 0xDD, 0xF5), // light blue
    (0.300, 0x87, 0xCE, 0xEB), // sky blue
    (0.400, 0x52, 0xA5, 0xD9), // steel blue
    (0.500, 0x31, 0x7F, 0xCA), // medium blue
    (0.600, 0x1E, 0x5A, 0xB0), // dark blue
    (0.700, 0x15, 0x3E, 0x90), // navy
    (0.800, 0x48, 0x2E, 0x8E), // purple
    (0.900, 0x6B, 0x24, 0x98), // dark purple
    (1.000, 0x9C, 0x27, 0xB0), // bright purple (extreme)
];

// ============================================================
// Ice accretion colormap: cyan/teal scale
// Designed for 0-2+ inches
// ============================================================
pub static ICE: &[ColorStop] = &[
    (0.000, 0xF0, 0xFF, 0xFF), // azure white
    (0.125, 0xCC, 0xF5, 0xF0), // very pale teal
    (0.250, 0x99, 0xE6, 0xDA), // light teal
    (0.375, 0x66, 0xCC, 0xCC), // medium teal
    (0.500, 0x33, 0xAA, 0xAA), // teal
    (0.625, 0x00, 0x88, 0x88), // dark teal
    (0.750, 0x00, 0x66, 0x66), // deeper teal
    (0.875, 0x00, 0x44, 0x55), // very dark teal
    (1.000, 0x00, 0x2B, 0x36), // near-black teal
];

// ============================================================
// Visibility colormap: red (low) -> yellow -> green (high)
// Designed for 0-10+ miles
// ============================================================
pub static VISIBILITY: &[ColorStop] = &[
    (0.000, 0x8B, 0x00, 0x00), // dark red (near-zero vis)
    (0.100, 0xCC, 0x00, 0x00), // red
    (0.200, 0xFF, 0x45, 0x00), // orange-red
    (0.300, 0xFF, 0x8C, 0x00), // dark orange
    (0.400, 0xFF, 0xC1, 0x07), // amber
    (0.500, 0xFF, 0xEB, 0x3B), // yellow
    (0.600, 0xCD, 0xDC, 0x39), // yellow-green
    (0.700, 0x8B, 0xC3, 0x4A), // light green
    (0.800, 0x4C, 0xAF, 0x50), // green
    (0.900, 0x2E, 0x7D, 0x32), // dark green
    (1.000, 0x1B, 0x5E, 0x20), // very dark green (unlimited vis)
];

// ============================================================
// Cloud cover colormap: white (clear) -> gray -> dark (overcast)
// Designed for 0-100%
// ============================================================
pub static CLOUD_COVER: &[ColorStop] = &[
    (0.000, 0xFF, 0xFF, 0xFF), // white (clear sky)
    (0.125, 0xF0, 0xF0, 0xF0),
    (0.250, 0xD0, 0xD0, 0xD0),
    (0.375, 0xB0, 0xB0, 0xB0),
    (0.500, 0x90, 0x90, 0x90),
    (0.625, 0x70, 0x70, 0x70),
    (0.750, 0x55, 0x55, 0x55),
    (0.875, 0x3A, 0x3A, 0x3A),
    (1.000, 0x20, 0x20, 0x20), // near-black (total overcast)
];

// ============================================================
// Helicity (SRH) colormap: diverging purple-white-orange
// Designed for -500 to +500 m2/s2
// ============================================================
pub static HELICITY: &[ColorStop] = &[
    (0.000, 0x4A, 0x00, 0x82), // dark purple (strong negative)
    (0.125, 0x6A, 0x1B, 0x9A),
    (0.250, 0x9C, 0x4D, 0xCC),
    (0.375, 0xCE, 0x93, 0xD8),
    (0.450, 0xE8, 0xD0, 0xF0),
    (0.500, 0xF7, 0xF7, 0xF7), // white (zero)
    (0.550, 0xFD, 0xE0, 0xC0),
    (0.625, 0xFD, 0xAE, 0x61),
    (0.750, 0xF4, 0x6D, 0x0F),
    (0.875, 0xD9, 0x48, 0x01),
    (1.000, 0x8C, 0x2D, 0x04), // dark orange (strong positive)
];

// ============================================================
// Divergence colormap: blue(convergence)-white-red(divergence)
// ============================================================
pub static DIVERGENCE: &[ColorStop] = &[
    (0.000, 0x08, 0x30, 0x6B), // deep blue (strong convergence)
    (0.125, 0x21, 0x66, 0xAC),
    (0.250, 0x4B, 0x96, 0xD0),
    (0.375, 0x92, 0xC5, 0xDE),
    (0.450, 0xD1, 0xE5, 0xF0),
    (0.500, 0xF7, 0xF7, 0xF7), // white (zero)
    (0.550, 0xF4, 0xCA, 0xC0),
    (0.625, 0xEF, 0x8A, 0x62),
    (0.750, 0xDA, 0x4E, 0x2B),
    (0.875, 0xB2, 0x18, 0x2B),
    (1.000, 0x67, 0x00, 0x1F), // deep red (strong divergence)
];

// ============================================================
// Theta-E (equivalent potential temperature) colormap
// Similar to temperature but shifted for theta-e range (280-380K)
// ============================================================
pub static THETA_E: &[ColorStop] = &[
    (0.000, 0x08, 0x30, 0x6B), // deep blue (cold/dry air)
    (0.100, 0x21, 0x66, 0xAC),
    (0.200, 0x4B, 0x96, 0xD0),
    (0.300, 0x92, 0xC5, 0xDE),
    (0.400, 0xBE, 0xDD, 0x72), // yellow-green
    (0.500, 0xFE, 0xEB, 0x65), // yellow
    (0.600, 0xFD, 0xAE, 0x61), // orange
    (0.700, 0xF4, 0x6D, 0x43), // red-orange
    (0.800, 0xD7, 0x30, 0x27), // red
    (0.900, 0xA5, 0x00, 0x26), // dark red
    (1.000, 0x67, 0x00, 0x1F), // very dark red (warm/moist)
];

// ============================================================
// NWS Reflectivity colormap: exact NWS radar color table
// <5 dBZ transparent, 5-75 dBZ standard NWS colors
// ============================================================
pub static NWS_REFLECTIVITY: &[ColorStop] = &[
    (0.000, 0x00, 0x00, 0x00), // <5 dBZ: no echo (use as transparent)
    (0.067, 0x00, 0x00, 0x00), // 5 dBZ boundary
    (0.070, 0x04, 0xE9, 0xE7), // light cyan-green
    (0.143, 0x01, 0xA0, 0x14), // green (20 dBZ)
    (0.214, 0x00, 0xC8, 0x00), // bright green
    (0.286, 0x02, 0xEB, 0x02), // light green (25 dBZ)
    (0.357, 0xFA, 0xFB, 0x00), // yellow (30 dBZ)
    (0.429, 0xEB, 0xCF, 0x00), // dark yellow (35 dBZ)
    (0.500, 0xFF, 0x9C, 0x00), // orange (40 dBZ)
    (0.571, 0xFF, 0x00, 0x00), // red (45 dBZ)
    (0.643, 0xD4, 0x00, 0x00), // dark red (50 dBZ)
    (0.714, 0xC0, 0x00, 0x00), // darker red (55 dBZ)
    (0.786, 0xFF, 0x00, 0xF0), // magenta (60 dBZ)
    (0.857, 0x98, 0x54, 0xC6), // purple (65 dBZ)
    (0.929, 0xFF, 0xFF, 0xFF), // white (70+ dBZ)
    (1.000, 0xE0, 0xE0, 0xE0), // light gray (75+ dBZ)
];

// ============================================================
// NWS QPE Precipitation colormap: standard NWS quantitative precip
// Light green -> dark green -> yellow -> orange -> red -> purple
// ============================================================
pub static NWS_PRECIP: &[ColorStop] = &[
    (0.000, 0xAD, 0xFF, 0xAD), // very light green (trace)
    (0.083, 0x7E, 0xFF, 0x7E), // light green
    (0.167, 0x2B, 0xCB, 0x2B), // green (0.25")
    (0.250, 0x00, 0x8B, 0x00), // dark green (0.50")
    (0.333, 0xFF, 0xFF, 0x00), // yellow (0.75")
    (0.417, 0xFF, 0xD7, 0x00), // gold (1.0")
    (0.500, 0xFF, 0x9C, 0x00), // orange (1.5")
    (0.583, 0xFF, 0x57, 0x00), // dark orange (2.0")
    (0.667, 0xFF, 0x00, 0x00), // red (2.5")
    (0.750, 0xCC, 0x00, 0x00), // dark red (3.0")
    (0.833, 0xAA, 0x00, 0x50), // crimson (4.0")
    (0.917, 0x88, 0x00, 0x88), // purple (5.0")
    (1.000, 0xFF, 0x80, 0xFF), // light magenta (6.0"+)
];

// ============================================================
// GOES IR Enhancement colormap: satellite infrared
// Warm=black/gray, cold=white with color-enhanced cloud tops
// ============================================================
pub static GOES_IR: &[ColorStop] = &[
    (0.000, 0x00, 0x00, 0x00), // black (warm surface, ~300K)
    (0.150, 0x3C, 0x3C, 0x3C), // dark gray
    (0.300, 0x78, 0x78, 0x78), // medium gray
    (0.400, 0xB0, 0xB0, 0xB0), // light gray
    (0.450, 0xD0, 0xD0, 0xD0), // very light gray
    (0.500, 0xFF, 0xFF, 0xFF), // white (cloud tops ~240K)
    (0.550, 0x00, 0xCC, 0xFF), // cyan (enhanced cold tops)
    (0.600, 0x00, 0x66, 0xFF), // blue
    (0.650, 0xFF, 0xFF, 0x00), // yellow
    (0.700, 0xFF, 0x99, 0x00), // orange
    (0.750, 0xFF, 0x00, 0x00), // red
    (0.800, 0xCC, 0x00, 0x66), // magenta
    (0.850, 0x99, 0x00, 0xCC), // purple (very cold overshooting tops)
    (0.900, 0x33, 0x00, 0x66), // dark purple
    (1.000, 0x00, 0x00, 0x00), // black (coldest, ~170K)
];

// ============================================================
// NWS-style temperature colormap (classic NWS GFE colors)
// Deep purple/blue < -20F through dark red/magenta > 100F
// Designed for -40F to 120F
// ============================================================
pub static TEMPERATURE_NWS: &[ColorStop] = &[
    (0.000, 0x4B, 0x00, 0x82), // deep purple (-40F)
    (0.063, 0x3A, 0x00, 0x9E), // purple (-30F)
    (0.125, 0x00, 0x00, 0xCC), // deep blue (-20F)
    (0.188, 0x00, 0x44, 0xEE), // blue (-10F)
    (0.250, 0x00, 0x88, 0xFF), // medium blue (0F)
    (0.313, 0x00, 0xBB, 0xFF), // cyan-blue (10F)
    (0.375, 0x00, 0xDD, 0xDD), // cyan (20F)
    (0.438, 0x00, 0xCC, 0x66), // teal-green (30F)
    (0.500, 0x00, 0xAA, 0x00), // green (40F)
    (0.531, 0x66, 0xCC, 0x00), // yellow-green (45F)
    (0.563, 0xCC, 0xDD, 0x00), // yellow (50F)
    (0.625, 0xFF, 0xEE, 0x00), // bright yellow (60F)
    (0.688, 0xFF, 0xBB, 0x00), // orange-yellow (70F)
    (0.750, 0xFF, 0x88, 0x00), // orange (80F)
    (0.813, 0xFF, 0x44, 0x00), // red-orange (90F)
    (0.875, 0xEE, 0x00, 0x00), // red (100F)
    (0.938, 0xCC, 0x00, 0x44), // dark red (110F)
    (1.000, 0xAA, 0x00, 0x88), // magenta (120F)
];

// ============================================================
// Professional style temperature (muted tones)
// Cool blues through greens through warm oranges/reds
// ============================================================
pub static TEMPERATURE_PIVOTAL: &[ColorStop] = &[
    (0.000, 0x1A, 0x23, 0x5B), // dark navy (-40F)
    (0.083, 0x2B, 0x4B, 0x8A), // dark blue (-20F)
    (0.167, 0x3D, 0x7E, 0xAE), // steel blue (0F)
    (0.250, 0x5B, 0xA8, 0xC8), // muted cyan (10F)
    (0.333, 0x8C, 0xCA, 0xBB), // sage (25F)
    (0.400, 0x6D, 0xA8, 0x6D), // muted green (35F)
    (0.450, 0xA3, 0xBF, 0x6F), // olive green (40F)
    (0.500, 0xCC, 0xD3, 0x72), // khaki (50F)
    (0.556, 0xE8, 0xD5, 0x6B), // warm yellow (55F)
    (0.611, 0xE8, 0xB4, 0x5C), // muted gold (65F)
    (0.667, 0xDD, 0x8E, 0x50), // muted orange (70F)
    (0.722, 0xCC, 0x66, 0x44), // terra cotta (80F)
    (0.778, 0xB8, 0x44, 0x3B), // muted red (85F)
    (0.833, 0xA0, 0x2C, 0x3C), // dark red (95F)
    (0.889, 0x82, 0x1C, 0x40), // maroon (105F)
    (0.944, 0x6A, 0x14, 0x48), // dark maroon (110F)
    (1.000, 0x52, 0x10, 0x4A), // deep purple-brown (120F)
];

// ============================================================
// CAPE colormap (professional meteorological style)
// Gray/white -> yellow -> orange -> red -> magenta/purple
// Designed for 0-5000+ J/kg
// ============================================================
pub static CAPE_PIVOTAL: &[ColorStop] = &[
    (0.000, 0xF5, 0xF5, 0xF5), // near-white (0)
    (0.050, 0xD0, 0xD0, 0xD0), // light gray (250)
    (0.100, 0xAA, 0xAA, 0xAA), // gray (500)
    (0.200, 0xF0, 0xE6, 0x8C), // light khaki (1000)
    (0.300, 0xF0, 0xC8, 0x40), // gold (1500)
    (0.400, 0xE8, 0x99, 0x20), // orange (2000)
    (0.500, 0xE0, 0x60, 0x10), // red-orange (2500)
    (0.600, 0xCC, 0x20, 0x10), // red (3000)
    (0.700, 0xB0, 0x00, 0x30), // crimson (3500)
    (0.800, 0xA0, 0x00, 0x70), // magenta (4000)
    (0.900, 0x80, 0x00, 0x90), // purple (4500)
    (1.000, 0x60, 0x00, 0xA0), // deep purple (5000+)
];

// ============================================================
// Wind speed colormap (professional meteorological style)
// White/light -> green -> yellow -> orange -> red -> purple
// Designed for 0-60 kt
// ============================================================
pub static WIND_PIVOTAL: &[ColorStop] = &[
    (0.000, 0xF8, 0xF8, 0xF8), // near-white (calm)
    (0.083, 0xD0, 0xE8, 0xD0), // very light green (5kt)
    (0.167, 0x70, 0xC0, 0x70), // light green (10kt)
    (0.250, 0x30, 0x99, 0x30), // green (15kt)
    (0.333, 0x20, 0x77, 0x20), // dark green (20kt)
    (0.417, 0xCC, 0xCC, 0x00), // yellow (25kt)
    (0.500, 0xE8, 0xAA, 0x00), // gold (30kt)
    (0.583, 0xF0, 0x80, 0x00), // orange (35kt)
    (0.667, 0xE8, 0x44, 0x00), // red-orange (40kt)
    (0.750, 0xDD, 0x00, 0x00), // red (45kt)
    (0.833, 0xBB, 0x00, 0x44), // crimson (50kt)
    (0.917, 0x99, 0x00, 0x88), // magenta (55kt)
    (1.000, 0x77, 0x00, 0xAA), // purple (60kt+)
];

// ============================================================
// Reflectivity (clean style for dark map backgrounds)
// Adjusted NWS colors with better contrast on dark tiles
// ============================================================
pub static REFLECTIVITY_CLEAN: &[ColorStop] = &[
    (0.000, 0x00, 0x00, 0x00), // transparent/black (<5 dBZ)
    (0.067, 0x00, 0x00, 0x00), // 5 dBZ boundary
    (0.070, 0x00, 0x70, 0x70), // dark teal (5 dBZ)
    (0.143, 0x00, 0x99, 0x44), // dark green (15 dBZ)
    (0.214, 0x00, 0xBB, 0x00), // green (20 dBZ)
    (0.286, 0x44, 0xDD, 0x00), // bright green (25 dBZ)
    (0.357, 0xDD, 0xDD, 0x00), // yellow (30 dBZ)
    (0.429, 0xDD, 0xAA, 0x00), // dark yellow (35 dBZ)
    (0.500, 0xEE, 0x77, 0x00), // orange (40 dBZ)
    (0.571, 0xEE, 0x22, 0x00), // red (45 dBZ)
    (0.643, 0xBB, 0x00, 0x00), // dark red (50 dBZ)
    (0.714, 0x99, 0x00, 0x00), // darker red (55 dBZ)
    (0.786, 0xEE, 0x00, 0xDD), // bright magenta (60 dBZ)
    (0.857, 0x88, 0x44, 0xBB), // purple (65 dBZ)
    (0.929, 0xEE, 0xEE, 0xEE), // bright white (70+ dBZ)
    (1.000, 0xCC, 0xCC, 0xCC), // light gray (75+ dBZ)
];

/// Look up a named colormap. Returns None if the name is not recognized.
pub fn get_colormap(name: &str) -> Option<&'static [ColorStop]> {
    match name {
        "temperature" | "temp" | "temperature_f" | "temperature_c"
            | "temperature_250" | "temperature_500" | "temperature_700"
            => Some(TEMPERATURE),
        "dewpoint" | "dewpoint_f" | "dewpoint_c" => Some(DEWPOINT),
        "precipitation" | "precip" | "precip_in" | "rain" => Some(PRECIPITATION),
        "wind" | "winds" | "winds_sfc" => Some(WIND),
        "reflectivity" | "refl" | "dbz" => Some(REFLECTIVITY),
        "cape" | "three_cape" | "stp" | "ehi" | "uh"
            | "lapse_rate" | "ml_metric" => Some(CAPE),
        "relative_humidity" | "rh" => Some(RELATIVE_HUMIDITY),
        "vorticity" | "relvort" | "geopot_anomaly" => Some(VORTICITY),
        "pressure" | "mslp" | "altimeter" => Some(PRESSURE),
        "snow" | "snow_accum" | "snowfall" => Some(SNOW),
        "ice" | "ice_accum" | "freezing_rain" => Some(ICE),
        "visibility" | "vis" => Some(VISIBILITY),
        "cloud_cover" | "cloud" | "clouds" | "tcc" => Some(CLOUD_COVER),
        "helicity" | "srh" => Some(HELICITY),
        "divergence" | "div" | "convergence" => Some(DIVERGENCE),
        "theta_e" | "thetae" | "equivalent_potential_temperature" => Some(THETA_E),
        "nws_reflectivity" | "nws_radar" | "nexrad" => Some(NWS_REFLECTIVITY),
        "nws_precip" | "nws_qpe" | "qpe" => Some(NWS_PRECIP),
        "goes_ir" | "satellite_ir" | "ir" => Some(GOES_IR),
        "temperature_nws" | "temp_nws" => Some(TEMPERATURE_NWS),
        "temperature_pivotal" | "temp_pivotal" => Some(TEMPERATURE_PIVOTAL),
        "cape_pivotal" => Some(CAPE_PIVOTAL),
        "wind_pivotal" | "winds_pivotal" => Some(WIND_PIVOTAL),
        "reflectivity_clean" | "refl_clean" => Some(REFLECTIVITY_CLEAN),
        _ => None,
    }
}

/// List all available colormap names.
pub fn list_colormaps() -> &'static [&'static str] {
    &[
        "temperature",
        "dewpoint",
        "precipitation",
        "wind",
        "reflectivity",
        "cape",
        "relative_humidity",
        "vorticity",
        "pressure",
        "snow",
        "ice",
        "visibility",
        "cloud_cover",
        "helicity",
        "divergence",
        "theta_e",
        "nws_reflectivity",
        "nws_precip",
        "goes_ir",
        "temperature_nws",
        "temperature_pivotal",
        "cape_pivotal",
        "wind_pivotal",
        "reflectivity_clean",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate_endpoints() {
        let (r, g, b) = interpolate_color(TEMPERATURE, 0.0);
        assert_eq!((r, g, b), (0x2b, 0x5d, 0x7e));

        let (r, g, b) = interpolate_color(TEMPERATURE, 1.0);
        assert_eq!((r, g, b), (0x45, 0x48, 0x44));
    }

    #[test]
    fn test_interpolate_midpoint() {
        let (r, g, b) = interpolate_color(TEMPERATURE, 0.5);
        // Should be the control point at 0.500
        assert_eq!((r, g, b), (0x0f, 0x55, 0x75));
    }

    #[test]
    fn test_clamp_out_of_range() {
        let below = interpolate_color(TEMPERATURE, -0.5);
        let at_zero = interpolate_color(TEMPERATURE, 0.0);
        assert_eq!(below, at_zero);

        let above = interpolate_color(TEMPERATURE, 1.5);
        let at_one = interpolate_color(TEMPERATURE, 1.0);
        assert_eq!(above, at_one);
    }

    #[test]
    fn test_get_colormap() {
        assert!(get_colormap("temperature").is_some());
        assert!(get_colormap("wind").is_some());
        assert!(get_colormap("nonexistent").is_none());
    }

    #[test]
    fn test_styled_colormaps_exist() {
        // All styled colormaps should be resolvable
        assert!(get_colormap("temperature_nws").is_some());
        assert!(get_colormap("temperature_pivotal").is_some());
        assert!(get_colormap("cape_pivotal").is_some());
        assert!(get_colormap("wind_pivotal").is_some());
        assert!(get_colormap("reflectivity_clean").is_some());
        assert!(get_colormap("temp_nws").is_some());
        assert!(get_colormap("temp_pivotal").is_some());
        assert!(get_colormap("refl_clean").is_some());
    }

    #[test]
    fn test_styled_colormaps_interpolate() {
        // Each styled colormap should produce valid colors at endpoints and midpoint
        let styled = [
            TEMPERATURE_NWS, TEMPERATURE_PIVOTAL, CAPE_PIVOTAL,
            WIND_PIVOTAL, REFLECTIVITY_CLEAN,
        ];
        for cmap in &styled {
            let (r0, g0, b0) = interpolate_color(cmap, 0.0);
            let (r5, g5, b5) = interpolate_color(cmap, 0.5);
            let (r1, g1, b1) = interpolate_color(cmap, 1.0);
            // Just verify they don't panic and produce different colors at extremes
            assert!(r0 != r1 || g0 != g1 || b0 != b1, "Endpoints should differ");
        }
    }

    #[test]
    fn test_all_listed_colormaps_resolve() {
        // Every name in list_colormaps() should resolve via get_colormap()
        for name in list_colormaps() {
            assert!(
                get_colormap(name).is_some(),
                "Listed colormap '{}' should resolve via get_colormap()",
                name,
            );
        }
    }
}
