//! Weather-focused colormap anchor tables used by the Rust renderer.

use crate::color::{Rgba, lerp_hex};

// -----------------------------------------------------------------------
// Raw anchor data (hex strings).
// -----------------------------------------------------------------------

const WINDS: &[&str] = &[
    "#ffffff", "#87cefa", "#6a5acd", "#e696dc", "#c85abe", "#a01496", "#c80028", "#dc283c",
    "#f05050", "#faf064", "#dcbe46", "#be8c28", "#a05a0a",
];

/// The fixed temperature ramp: 19 anchors evenly spaced over -60..120 F,
/// one every ten degrees, cropped by [`temperature_cropped`].
///
/// THE RULE THIS TABLE IS HELD TO: an anchor may be neutral ink only
/// when both of its neighbours carry chroma.  A neutral anchor is a
/// legible boundary between two hue families -- `#f7f7ff` at 10 F
/// separates the violet band below from the blue band above -- but a RUN
/// of them is a stretch of the range that carries no colour at all,
/// readable only by tone, on a table that spends neutral ink
/// elsewhere in the same range.
///
/// The top three anchors used to be `#e7e0da`, `#959391` and `#454844`:
/// a near-white and two greys, HSV saturation 0.056, 0.027 and 0.056,
/// descending in lightness (CIELAB L* 89.6, 61.0, 30.2).  The top
/// fifth of the range carried no chroma at all, so its structure was a
/// tone ramp rather than a colour one, the hottest ground came out a
/// dark grey below the near-white of the merely warm ground, and
/// against the near-white this table already draws at 10 F a pale cell
/// said nothing about which end of the range it came from.  On a 2 m
/// plate whose field ran 283 to 321 K, 38.53 percent of the cells
/// landed in that run.  They are a pink band now -- light pink,
/// vivid rose, deep violet -- a hue family no other part of the table
/// occupies.  The chroma is what makes the band read; it also descends
/// in lightness across its three anchors (CIELAB L* 70.8, 48.6, 21.0),
/// which is what gives it internal structure, a first pass at three
/// anchors of one lightness having come back as a single flat wash.
/// Read off two plates of one frame with
/// `tools/measure_plate_hot_band.py`, on the 235,789 map cells the
/// repaired plate draws at or above 100 F: median HSV saturation 0.032
/// before and 0.925 after, CIELAB L* p5 to p95 52.8 to 83.5 before and
/// 40.9 to 64.2 after.  The repaired band spends slightly LESS tone
/// than the run it replaced and reads anyway, because what it carries
/// is chroma.
/// The sixteen anchors below 100 F are unchanged, byte for byte, so the
/// cold end and every crop that stops at or under 90 F draw exactly what
/// they drew before.
///
/// The range stays FIXED at -60..120 F.  Every product in this family is
/// fixed-range -- dewpoint, relative humidity, wind speed, reflectivity,
/// accumulated precipitation -- because one frame of a series has to be
/// comparable with the next, which a range fitted to each frame's own
/// data is not.
const TEMPERATURE: &[&str] = &[
    "#2b5d7e", "#75a8b0", "#aee3dc", "#a0b8d6", "#968bc5", "#8243b2", "#a343b3", "#f7f7ff",
    "#a0b8d6", "#0f5575", "#6d8c77", "#f8eea2", "#aa714d", "#5f0000", "#852c40", "#b28f85",
    "#ff85c2", "#e0007a", "#5c0066",
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

/// The infrared ramp outside its enhanced window.  A brightness
/// temperature scale is a GREY scale by convention: bright for cold
/// cloud top, dark for warm ground, with one enhanced window in the
/// middle carrying the colour.
const SIM_IR_HIGH_CLOUD: &[&str] = &["#ffffff", "#d8d8d8", "#8a8a8a", "#000000"];
const SIM_IR_ENHANCED: &[&str] = &[
    "#000000", "#5b0000", "#fd0100", "#ff7f00", "#fcff05", "#03fd03", "#00651f", "#010077",
    "#0ff6ef",
];
const SIM_IR_MID_CLOUD: &[&str] = &["#ffffff", "#d5d5d5", "#a4a4a4"];
const SIM_IR_GROUND: &[&str] = &["#9a9a9a", "#606060", "#242424", "#000000"];
const SIM_IR_WARMEST: &[&str] = &["#000000", "#000000"];

/// The top of the composite ladder, shared by the two composites that
/// run past the segments above: near-black greys into a cyan cap, so
/// the extreme end of the scale reads as extreme rather than as one
/// more hue.
const COMP_TAIL: &[&str] = &[
    "#806a70", "#535057", "#20242a", "#31535a", "#55a3aa", "#83edf2",
];

/// The significant-tornado composite's own ladder: the composite base
/// segments below 4, then its own segments through the dark greys of
/// [`COMP_TAIL`] to the same cyan cap.
const STP_SEGS: &[(&[&str], usize)] = &[
    (COMP_SEG0, 5),
    (COMP_SEG1, 5),
    (COMP_SEG2, 5),
    (COMP_SEG3, 5),
    (&["#73088a", "#9e3fba", "#d992df", "#e9bec3"], 5),
    (&["#e9bec3", "#cf8f99", "#a95d69"], 5),
    (&["#a95d69", "#93606b", "#806a70"], 10),
    (&["#806a70", "#6a6066", "#535057"], 10),
    (&["#535057", "#403f46", "#2d3034"], 10),
    (&["#2d3034", "#20242a", "#15191d", "#20252a"], 10),
    (&["#20252a", "#263941", "#31535a"], 10),
    (&["#31535a", "#426f76", "#55a3aa", "#83edf2"], 20),
];

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
    c.extend(lerp_hex(SIM_IR_HIGH_CLOUD, 10)); // -80..-70
    c.extend(lerp_hex(SIM_IR_ENHANCED, 50)); // -70..-20
    c.extend(lerp_hex(SIM_IR_MID_CLOUD, 20)); // -20..0
    c.extend(lerp_hex(SIM_IR_GROUND, 40)); // 0..40
    c.extend(lerp_hex(SIM_IR_WARMEST, 10)); // 40..50
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
    c.extend(lerp_hex(COMP_TAIL, 50));
    c
}

/// SRH composite palette.
pub fn srh() -> Vec<Rgba> {
    let mut c = build_composite(&[10, 10, 10, 10, 10, 25, 20]);
    c.extend(lerp_hex(COMP_TAIL, 45));
    c
}

/// STP composite palette.
pub fn stp() -> Vec<Rgba> {
    build_segments(STP_SEGS)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Below this, ink is neutral: a grey, a near-white, a near-black.
    const NEUTRAL: f64 = 0.10;

    /// HSV saturation of one `#rrggbb` anchor.
    fn saturation(hex: &str) -> f64 {
        let channel = |start: usize| {
            u8::from_str_radix(&hex[start..start + 2], 16).expect("hex channel") as f64
        };
        let (r, g, b) = (channel(1), channel(3), channel(5));
        let high = r.max(g).max(b);
        let low = r.min(g).min(b);
        if high == 0.0 { 0.0 } else { (high - low) / high }
    }

    /// Euclidean distance between two `#rrggbb` anchors, in RGB.
    fn distance(left: &str, right: &str) -> f64 {
        let channel = |hex: &str, start: usize| {
            u8::from_str_radix(&hex[start..start + 2], 16).expect("hex channel")
                as f64
        };
        [1usize, 3, 5]
            .into_iter()
            .map(|start| (channel(left, start) - channel(right, start)).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// The anchor index one Fahrenheit value sits on.
    fn anchor_of(fahrenheit: f64) -> usize {
        (((fahrenheit + 60.0) / 180.0) * (TEMPERATURE.len() - 1) as f64).round() as usize
    }

    #[test]
    fn no_run_of_neutral_anchors_leaves_a_band_of_the_range_colourless() {
        // A single neutral anchor is a boundary between two hue families
        // and is allowed.  Two in a row is a stretch of the range a
        // reader cannot separate, which is what made every value at or
        // above 100 F read as grey.
        let neutral: Vec<usize> = (0..TEMPERATURE.len())
            .filter(|index| saturation(TEMPERATURE[*index]) < NEUTRAL)
            .collect();
        for pair in neutral.windows(2) {
            assert!(
                pair[1] - pair[0] > 1,
                "anchors {} and {} are both neutral ink",
                pair[0],
                pair[1]
            );
        }
        // And the hot end specifically: the band a heat plate lives in.
        // Well clear of neutral, and each step far enough from the one
        // below it to be a step -- three saturated anchors of one
        // lightness read as a single wash, which is unreadable in a
        // second way.
        for fahrenheit in [100.0, 110.0, 120.0] {
            let hex = TEMPERATURE[anchor_of(fahrenheit)];
            assert!(
                saturation(hex) >= 0.35,
                "{fahrenheit} F anchor {hex} carries too little chroma"
            );
            let below = TEMPERATURE[anchor_of(fahrenheit) - 1];
            assert!(
                distance(below, hex) >= 60.0,
                "{fahrenheit} F anchor {hex} does not read apart from {below} \
                 ten degrees below it"
            );
        }
    }

    #[test]
    fn every_crop_this_tree_asks_for_ends_on_chromatic_ink() {
        // Every crop the product tables ask for, read off
        // `rustwx-products/src/plot_design.rs`: the 2 m plate, the
        // fallback arm, the six isobaric arms and the two windowed 2 m
        // scales come to these six.  Each one's last colour is the one an
        // over-range value is drawn in, so a neutral there is the same
        // defect at the edge of a narrower product.
        for crop in [
            (-60.0, 120.0),
            (-40.0, 120.0),
            (-40.0, 110.0),
            (-40.0, 90.0),
            (-40.0, 70.0),
            (32.0, 110.0),
        ] {
            let colors = temperature_cropped(64, Some(crop));
            let last = colors.last().copied().expect("a cropped ramp has colours");
            let high = last.r.max(last.g).max(last.b) as f64;
            let low = last.r.min(last.g).min(last.b) as f64;
            let sat = if high == 0.0 { 0.0 } else { (high - low) / high };
            assert!(
                sat >= NEUTRAL,
                "crop {crop:?} ends on neutral ink rgb({},{},{})",
                last.r,
                last.g,
                last.b
            );
        }
    }

    #[test]
    fn the_cold_sixteen_anchors_are_unchanged() {
        // Everything at or under 90 F is pinned: the hot-end repair is
        // not licence to redraw a band that reads correctly, and every
        // product whose crop stops there still draws what it drew.
        assert_eq!(
            &TEMPERATURE[..=anchor_of(90.0)],
            &[
                "#2b5d7e", "#75a8b0", "#aee3dc", "#a0b8d6", "#968bc5", "#8243b2", "#a343b3",
                "#f7f7ff", "#a0b8d6", "#0f5575", "#6d8c77", "#f8eea2", "#aa714d", "#5f0000",
                "#852c40", "#b28f85",
            ]
        );
    }

    /// Every anchor table this CRATE builds a product ramp from, in the
    /// order a ramp built from it puts them -- segmented tables
    /// flattened, because the last anchor of one segment and the first
    /// of the next are adjacent in the ramp the reader sees.
    ///
    /// The rule below is a rule about PRODUCT PALETTES, not about one
    /// file, so this reaches the eleven tables defined in `weather.rs`
    /// as well as the twelve here.  A guard that reads one file and
    /// reports the rule is a guard claiming coverage it does not have,
    /// which is how the hot end of the temperature ramp survived a
    /// palette test that passed.
    fn every_product_table() -> Vec<(&'static str, Vec<&'static str>)> {
        let flatten = |segments: &[(&[&'static str], usize)]| -> Vec<&'static str> {
            segments
                .iter()
                .flat_map(|(anchors, _)| anchors.iter().copied())
                .collect()
        };
        let chain = |segments: &[&[&'static str]]| -> Vec<&'static str> {
            segments.iter().flat_map(|s| s.iter().copied()).collect()
        };
        let mut dewpoint: Vec<&str> = DEWPOINT_DRY.to_vec();
        dewpoint.extend(flatten(DEWPOINT_MOIST));
        let mut tables = vec![
            ("temperature", TEMPERATURE.to_vec()),
            ("dewpoint", dewpoint),
            ("winds", WINDS.to_vec()),
            ("relative humidity", chain(&[RH_SEG1, RH_SEG2, RH_SEG3])),
            (
                "simulated infrared",
                chain(&[
                    SIM_IR_COOL,
                    SIM_IR_HIGH_CLOUD,
                    SIM_IR_ENHANCED,
                    SIM_IR_MID_CLOUD,
                    SIM_IR_GROUND,
                    SIM_IR_WARMEST,
                ]),
            ),
            ("geopotential anomaly", GEOPOT_ANOMALY.to_vec()),
            ("relative vorticity", RELVORT.to_vec()),
            ("reflectivity", REFLECTIVITY.to_vec()),
            ("accumulated precipitation", flatten(PRECIP_SEGS)),
            (
                "composite",
                chain(&[
                    COMP_SEG0, COMP_SEG1, COMP_SEG2, COMP_SEG3, COMP_SEG4, COMP_SEG5,
                    COMP_SEG6,
                ]),
            ),
            ("composite tail", COMP_TAIL.to_vec()),
            ("significant tornado parameter", flatten(STP_SEGS)),
        ];
        tables.extend(crate::weather::fixed_range_tables());
        tables
    }

    /// The tables that paint a run of neutral ink ON PURPOSE, each with
    /// the convention that makes it correct.  An exemption nobody wrote
    /// down is an exemption the next edit reads as a defect, so the test
    /// below also fails when one of these stops having a run: a stale
    /// exemption is deleted rather than carried.
    const A_NEUTRAL_RUN_IS_THE_CONVENTION: &[(&str, &str)] = &[
        (
            "relative vorticity",
            "half the table is a grey ramp because only one sign of \
             rotation is the significant one and the other half is \
             meant to recede",
        ),
        (
            "reflectivity",
            "white at the bottom is no echo, and the white through grey \
             at the top is the radar convention for the extreme dBZ a \
             storm reaches once in a season",
        ),
        (
            "accumulated precipitation",
            "the white and the grey steps are the trace band below the \
             lowest plotted accumulation",
        ),
        (
            "composite",
            "the first segment is the band below the lowest significant \
             value of every composite drawn on it, which is meant to \
             recede behind the coloured segments above",
        ),
        (
            "simulated infrared",
            "a brightness temperature scale is a GREY scale by \
             convention, bright for a cold cloud top and dark for warm \
             ground; the one window that carries colour is the -70 to \
             -20 C enhancement, so the greys on either side of it are \
             the convention rather than a gap in it",
        ),
        (
            "significant tornado parameter",
            "the base segment is the band below the lowest significant \
             value, as on every composite, and the upper segments run \
             through near-black greys into the cyan cap that means the \
             top of the scale",
        ),
        (
            "equilibrium level",
            "the two near-white anchors are the bottom of a sequential \
             ramp whose low end is the shallowest equilibrium level, \
             drawn to recede behind the violets above it; the run is at \
             the BOTTOM of this range and nothing at the top of it is \
             drawn in neutral ink",
        ),
    ];

    /// The two files this crate declares its anchor tables in, read as
    /// TEXT, so that the reach of the rule below is what the source
    /// declares rather than what a registry says about itself.
    const COLORMAP_SOURCE: &str = include_str!("colormaps.rs");
    const WEATHER_SOURCE: &str = include_str!("weather.rs");
    const PALETTE_SOURCE_FILES: &[(&str, &str)] = &[
        ("colormaps.rs", COLORMAP_SOURCE),
        ("weather.rs", WEATHER_SOURCE),
    ];

    /// Every file-level `const NAME: &[&str]` a palette source declares,
    /// segmented `&[(&[&str], usize)]` tables included.  Anchors written
    /// inline inside a function body are not a declared table and are
    /// not read here.
    fn declared_anchor_tables(source: &str) -> Vec<&str> {
        source
            .lines()
            .filter_map(|line| {
                let rest = line
                    .strip_prefix("const ")
                    .or_else(|| line.strip_prefix("pub const "))
                    .or_else(|| line.strip_prefix("pub(crate) const "))?;
                let (name, tail) = rest.split_once(':')?;
                let declared_type = tail.split('=').next()?;
                if !declared_type.contains("&[&str]") {
                    return None;
                }
                Some(name.trim())
            })
            .collect()
    }

    /// One function's text, from its signature to the brace that closes
    /// its body.
    fn function_text<'a>(source: &'a str, signature: &str) -> &'a str {
        let start = source
            .find(signature)
            .unwrap_or_else(|| panic!("{signature} is not in the file it is read from"));
        let body = &source[start..];
        let mut depth = 0usize;
        for (offset, character) in body.char_indices() {
            match character {
                '{' => depth += 1,
                '}' if depth <= 1 => return &body[..offset + 1],
                '}' => depth -= 1,
                _ => {}
            }
        }
        panic!("{signature} has no closing brace")
    }

    /// Whether `text` uses `name` as a whole identifier, so that a table
    /// whose name is the prefix of another table's is not read as
    /// present because of that other one.
    fn names_identifier(text: &str, name: &str) -> bool {
        let bytes = text.as_bytes();
        let is_word = |byte: u8| byte == b'_' || byte.is_ascii_alphanumeric();
        text.match_indices(name).any(|(at, _)| {
            (at == 0 || !is_word(bytes[at - 1]))
                && (at + name.len() == bytes.len() || !is_word(bytes[at + name.len()]))
        })
    }

    #[test]
    fn every_fixed_range_product_table_carries_no_neutral_run_it_cannot_account_for() {
        // The same rule, applied to every product table in the crate, so
        // one that grows a grey tail later fails here rather than in a
        // forecaster's eye.
        for (name, table) in every_product_table() {
            let neutral: Vec<usize> = (0..table.len())
                .filter(|index| saturation(table[*index]) < NEUTRAL)
                .collect();
            let runs: Vec<&[usize]> = neutral
                .windows(2)
                .filter(|pair| pair[1] - pair[0] == 1)
                .collect();
            match A_NEUTRAL_RUN_IS_THE_CONVENTION
                .iter()
                .find(|(exempt, _)| *exempt == name)
            {
                Some((_, reason)) => assert!(
                    !runs.is_empty(),
                    "{name} no longer carries the run it is exempt for ({reason}); \
                     delete the exemption rather than carry it"
                ),
                None => assert!(
                    runs.is_empty(),
                    "{name}: anchors {} and {} are both neutral ink",
                    runs[0][0],
                    runs[0][1]
                ),
            }
        }
        // And every exemption names a table that exists.
        for (exempt, _) in A_NEUTRAL_RUN_IS_THE_CONVENTION {
            assert!(
                every_product_table().iter().any(|(name, _)| name == exempt),
                "{exempt} is exempt from a rule it is not held to"
            );
        }
        // And the guard's reach is read off the two source files rather
        // than asserted about itself: every anchor table either file
        // DECLARES has to be named in one of the two registries, so a
        // table added to the crate and left out of a registry fails
        // here instead of being drawn with the rule never reaching it.
        let names: Vec<&str> = every_product_table()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        let mut unique = names.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), names.len(), "a table is named twice");
        let registries = format!(
            "{}{}",
            function_text(COLORMAP_SOURCE, "fn every_product_table("),
            function_text(WEATHER_SOURCE, "fn fixed_range_tables(")
        );
        for &(file, source) in PALETTE_SOURCE_FILES {
            for table in declared_anchor_tables(source) {
                assert!(
                    names_identifier(&registries, table),
                    "{file} declares the anchor table {table} and no registry names it"
                );
            }
        }
    }
}
