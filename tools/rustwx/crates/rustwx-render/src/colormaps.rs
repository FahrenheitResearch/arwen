//! Weather-focused colormap anchor tables used by the Rust renderer.

use crate::color::{Rgba, lerp_hex};

// -----------------------------------------------------------------------
// Raw anchor data (hex strings).
// -----------------------------------------------------------------------

const WINDS: &[&str] = &[
    "#ffffff", "#87cefa", "#6a5acd", "#e696dc", "#c85abe", "#a01496", "#c80028", "#dc283c",
    "#f05050", "#faf064", "#dcbe46", "#be8c28", "#a05a0a",
];

const TEMPERATURE: &[&str] = &[
    "#2b5d7e", "#75a8b0", "#aee3dc", "#a0b8d6", "#968bc5", "#8243b2", "#a343b3", "#f7f7ff",
    "#a0b8d6", "#0f5575", "#6d8c77", "#f8eea2", "#aa714d", "#5f0000", "#852c40", "#b28f85",
    "#e7e0da", "#959391", "#454844",
];

const DEWPOINT_DRY: &[&str] = &["#996f4f", "#4d4236", "#f2f2d8"];
const DEWPOINT_MOIST: &[(&[&str], usize)] = &[
    (&["#e3f3e6", "#64c461"], 10),
    (&["#32ae32", "#084d06"], 10),
    (&["#66a3ad", "#12292a"], 10),
    (&["#66679d", "#2b1e63"], 10),
    (&["#714270", "#a27382"], 10),
];

const RH_SEG1: &[&str] = &["#a5734d", "#382f28", "#6e6559", "#a59b8e", "#ddd1c3"];
const RH_SEG2: &[&str] = &["#c8d7c0", "#004a2f"];
const RH_SEG3: &[&str] = &["#004123", "#28588c"];

const RELVORT: &[&str] = &[
    "#323232", "#4d4d4d", "#707070", "#8A8A8A", "#a1a1a1", "#c0c0c0", "#d6d6d6", "#e5e5e5",
    "#ffffff", "#fdd244", "#fea000", "#f16702", "#da2422", "#ab029b", "#78008f", "#44008b",
    "#000160", "#244488", "#4f85b2", "#73cadb", "#91fffd",
];

const SIM_IR_COOL: &[&str] = &["#7f017f", "#e36fbe"];

// Composite base segments (shared by CAPE, SRH, STP, EHI, LR, UH, ML)
const COMP_SEG0: &[&str] = &["#ffffff", "#696969"];
const COMP_SEG1: &[&str] = &["#37536a", "#a7c8ce"];
const COMP_SEG2: &[&str] = &["#e9dd96", "#e16f02"];
const COMP_SEG3: &[&str] = &["#dc4110", "#8b0950"];
const COMP_SEG4: &[&str] = &["#73088a", "#da99e7"];
const COMP_SEG5: &[&str] = &["#e9bec3", "#b2445a"];
const COMP_SEG6: &[&str] = &["#893d48", "#bc9195"];

const REFLECTIVITY: &[&str] = &[
    "#ffffff", "#f2f6fc", "#d9e3f4", "#b0c6e6", "#8aa7da", "#648bcb", "#396dc1", "#1350b4",
    "#0d4f5d", "#43736f", "#77987b", "#a8bf8b", "#fdf273", "#f2d45a", "#eeb247", "#e1932d",
    "#d97517", "#cd5403", "#cd0002", "#a10206", "#75030b", "#9e37ab", "#83259d", "#601490",
    "#818181", "#b3b3b3", "#e8e8e8",
];

const GEOPOT_ANOMALY: &[&str] = &[
    "#c9f2fc", "#e684f4", "#732164", "#7b2b8d", "#8a41d6", "#253fba", "#7089cb", "#c0d5e8",
    "#ffffff", "#fbcfa1", "#fc984b", "#b83800", "#a3241a", "#5e1425", "#42293e", "#557b75",
    "#ddd5cf",
];

const PRECIP_SEGS: &[(&[&str], usize)] = &[
    (&["#ffffff", "#ffffff"], 1),
    (&["#dcdcdc", "#bebebe", "#9e9e9e", "#818181"], 9),
    (&["#b8f0c1", "#156471"], 40),
    (&["#164fba", "#d8edf5"], 50),
    (&["#cfbddd", "#a134b1"], 100),
    (&["#a43c32", "#dd9c98"], 200),
    (&["#f6f0a3", "#7e4b26", "#542f17"], 1100),
];

// -----------------------------------------------------------------------
// Composite builder (mirrors create_custom_cmap)
// -----------------------------------------------------------------------

fn build_composite(quants: &[usize; 7]) -> Vec<Rgba> {
    let segs: [&[&str]; 7] = [
        COMP_SEG0, COMP_SEG1, COMP_SEG2, COMP_SEG3, COMP_SEG4, COMP_SEG5, COMP_SEG6,
    ];
    let mut colors = Vec::new();
    for (seg, &n) in segs.iter().zip(quants.iter()) {
        if n > 0 {
            colors.extend(lerp_hex(seg, n));
        }
    }
    colors
}

fn build_segments(segs: &[(&[&str], usize)]) -> Vec<Rgba> {
    let mut colors = Vec::new();
    for (anchors, n) in segs {
        if *n > 0 {
            colors.extend(lerp_hex(anchors, *n));
        }
    }
    colors
}

// -----------------------------------------------------------------------
// Public palette constructors — return Vec<Rgba>
// -----------------------------------------------------------------------

/// 27-colour discrete reflectivity palette (ListedColormap).
pub fn reflectivity() -> Vec<Rgba> {
    REFLECTIVITY.iter().map(|h| Rgba::from_hex(h)).collect()
}

/// Winds palette (13 anchors → n segments).
pub fn winds(n: usize) -> Vec<Rgba> {
    lerp_hex(WINDS, n)
}

/// Temperature palette (19 anchors → n segments).
pub fn temperature(n: usize) -> Vec<Rgba> {
    lerp_hex(TEMPERATURE, n)
}

/// Temperature palette cropped using the upstream Fahrenheit-range slicing.
pub fn temperature_cropped(n: usize, crop_f: Option<(f64, f64)>) -> Vec<Rgba> {
    let anchors = if let Some((start, end)) = crop_f {
        let last = TEMPERATURE.len().saturating_sub(1) as f64;
        let start_index = (((start + 60.0) / 180.0) * last).floor() as usize;
        let end_index = (((end + 60.0) / 180.0) * last).floor() as usize;
        let start_index = start_index.min(TEMPERATURE.len().saturating_sub(1));
        let end_index = end_index.min(TEMPERATURE.len().saturating_sub(1));
        &TEMPERATURE[start_index..=end_index]
    } else {
        TEMPERATURE
    };
    lerp_hex(anchors, n)
}

/// Dewpoint palette (dry 80 + moist 5×10 = 130 segments).
pub fn dewpoint(dry: usize, moist_points_total: usize) -> Vec<Rgba> {
    let mut c = lerp_hex(DEWPOINT_DRY, dry);
    let moist_per_seg = moist_points_total / DEWPOINT_MOIST.len().max(1);
    for (anchors, _default_n) in DEWPOINT_MOIST {
        c.extend(lerp_hex(anchors, moist_per_seg));
    }
    c
}

/// Relative humidity palette (40 + 50 + 10 = 100 segments).
pub fn rh() -> Vec<Rgba> {
    let mut c = lerp_hex(RH_SEG1, 40);
    c.extend(lerp_hex(RH_SEG2, 50));
    c.extend(lerp_hex(RH_SEG3, 10));
    c
}

/// Relative vorticity palette (21 anchors → n segments).
pub fn relvort(n: usize) -> Vec<Rgba> {
    lerp_hex(RELVORT, n)
}

/// Simulated IR palette keyed to -90..50 C one-degree bins.
pub fn sim_ir() -> Vec<Rgba> {
    let mut c = lerp_hex(SIM_IR_COOL, 10); // -90..-80
    c.extend(lerp_hex(&["#ffffff", "#d8d8d8", "#8a8a8a", "#000000"], 10)); // -80..-70
    c.extend(lerp_hex(
        &[
            "#000000", "#5b0000", "#fd0100", "#ff7f00", "#fcff05", "#03fd03", "#00651f", "#010077",
            "#0ff6ef",
        ],
        50,
    )); // -70..-20
    c.extend(lerp_hex(&["#ffffff", "#d5d5d5", "#a4a4a4"], 20)); // -20..0
    c.extend(lerp_hex(&["#9a9a9a", "#606060", "#242424", "#000000"], 40)); // 0..40
    c.extend(lerp_hex(&["#000000", "#000000"], 10)); // 40..50
    c
}

/// CAPE composite palette.
pub fn cape() -> Vec<Rgba> {
    build_composite(&[10, 10, 10, 10, 10, 10, 20])
}

/// 0-3km CAPE composite palette.
pub fn three_cape() -> Vec<Rgba> {
    build_composite(&[10, 10, 10, 10, 10, 10, 40])
}

/// EHI composite palette.
pub fn ehi() -> Vec<Rgba> {
    let mut c = build_composite(&[10, 10, 20, 20, 20, 30, 20]);
    c.extend(lerp_hex(
        &[
            "#806a70", "#535057", "#20242a", "#31535a", "#55a3aa", "#83edf2",
        ],
        50,
    ));
    c
}

/// SRH composite palette.
pub fn srh() -> Vec<Rgba> {
    let mut c = build_composite(&[10, 10, 10, 10, 10, 25, 20]);
    c.extend(lerp_hex(
        &[
            "#806a70", "#535057", "#20242a", "#31535a", "#55a3aa", "#83edf2",
        ],
        45,
    ));
    c
}

/// STP composite palette.
pub fn stp() -> Vec<Rgba> {
    let mut c = lerp_hex(COMP_SEG0, 5); // 0-1
    c.extend(lerp_hex(COMP_SEG1, 5)); // 1-2
    c.extend(lerp_hex(COMP_SEG2, 5)); // 2-3
    c.extend(lerp_hex(COMP_SEG3, 5)); // 3-4
    c.extend(lerp_hex(&["#73088a", "#9e3fba", "#d992df", "#e9bec3"], 5)); // 4-5
    c.extend(lerp_hex(&["#e9bec3", "#cf8f99", "#a95d69"], 5)); // 5-6
    c.extend(lerp_hex(&["#a95d69", "#93606b", "#806a70"], 10)); // 6-8
    c.extend(lerp_hex(&["#806a70", "#6a6066", "#535057"], 10)); // 8-10
    c.extend(lerp_hex(&["#535057", "#403f46", "#2d3034"], 10)); // 10-12
    c.extend(lerp_hex(&["#2d3034", "#20242a", "#15191d", "#20252a"], 10)); // 12-14
    c.extend(lerp_hex(&["#20252a", "#263941", "#31535a"], 10)); // 14-16
    c.extend(lerp_hex(&["#31535a", "#426f76", "#55a3aa", "#83edf2"], 20)); // 16-20
    c
}

/// Lapse rate composite palette.
pub fn lapse_rate() -> Vec<Rgba> {
    build_composite(&[40, 10, 10, 10, 10, 0, 0])
}

/// Updraft helicity composite palette.
pub fn uh() -> Vec<Rgba> {
    build_composite(&[10, 10, 10, 10, 20, 20, 0])
}

/// ML metric composite palette.
pub fn ml_metric() -> Vec<Rgba> {
    build_composite(&[10, 10, 10, 10, 10, 10, 10])
}

/// Geopotential height anomaly palette (17 anchors → n segments).
pub fn geopot_anomaly(n: usize) -> Vec<Rgba> {
    lerp_hex(GEOPOT_ANOMALY, n)
}

/// Precipitation palette.
pub fn precip_in() -> Vec<Rgba> {
    build_segments(PRECIP_SEGS)
}

/// Shaded overlay: transparent black → semi-transparent black.
pub fn shaded_overlay() -> Vec<Rgba> {
    vec![
        Rgba::with_alpha(0, 0, 0, 0),
        Rgba::with_alpha(0, 0, 0, 0x60),
    ]
}
