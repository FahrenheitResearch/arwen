//! Radar color tables with precomputed LUT for fast rendering.

use crate::products::RadarProduct;

const LUT_SIZE: usize = 4096;

#[derive(Debug, Clone, Copy)]
struct ColorEntry { value: f32, r: u8, g: u8, b: u8, a: u8 }

#[derive(Debug, Clone)]
pub struct ColorTable {
    entries: Vec<ColorEntry>,
    min_value: f32,
    max_value: f32,
    lut: Vec<[u8; 4]>,
    lut_scale: f32,
}

fn ce(value: f32, r: u8, g: u8, b: u8, a: u8) -> ColorEntry {
    ColorEntry { value, r, g, b, a }
}

fn interpolate_entries(entries: &[ColorEntry], value: f32) -> [u8; 4] {
    if entries.is_empty() { return [0, 0, 0, 0]; }
    if value <= entries[0].value { let e = &entries[0]; return [e.r, e.g, e.b, e.a]; }
    let last = entries.len() - 1;
    if value >= entries[last].value { let e = &entries[last]; return [e.r, e.g, e.b, e.a]; }
    let mut lo = 0usize;
    let mut hi = last;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if entries[mid].value <= value { lo = mid; } else { hi = mid; }
    }
    let lower = &entries[lo];
    let upper = &entries[hi];
    let span = upper.value - lower.value;
    if span.abs() < 1e-6 { return [lower.r, lower.g, lower.b, lower.a]; }
    let t = ((value - lower.value) / span).clamp(0.0, 1.0);
    [
        (lower.r as f32 + t * (upper.r as f32 - lower.r as f32)) as u8,
        (lower.g as f32 + t * (upper.g as f32 - lower.g as f32)) as u8,
        (lower.b as f32 + t * (upper.b as f32 - lower.b as f32)) as u8,
        (lower.a as f32 + t * (upper.a as f32 - lower.a as f32)) as u8,
    ]
}

impl ColorTable {
    fn from_entries(mut entries: Vec<ColorEntry>) -> Self {
        entries.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(std::cmp::Ordering::Equal));
        let min_value = entries.first().map(|e| e.value).unwrap_or(0.0);
        let max_value = entries.last().map(|e| e.value).unwrap_or(1.0);
        let range = max_value - min_value;
        let lut_scale = if range > 0.0 { (LUT_SIZE - 1) as f32 / range } else { 1.0 };
        let mut lut = Vec::with_capacity(LUT_SIZE);
        for i in 0..LUT_SIZE {
            let value = min_value + (i as f32) / lut_scale;
            lut.push(interpolate_entries(&entries, value));
        }
        ColorTable { entries, min_value, max_value, lut, lut_scale }
    }

    #[inline]
    pub fn color_for_value(&self, value: f32) -> [u8; 4] {
        if value.is_nan() || value < self.min_value { return [0, 0, 0, 0]; }
        if value >= self.max_value {
            if let Some(last) = self.entries.last() { return [last.r, last.g, last.b, last.a]; }
            return [0, 0, 0, 0];
        }
        let idx = ((value - self.min_value) * self.lut_scale) as usize;
        self.lut[idx.min(LUT_SIZE - 1)]
    }

    pub fn for_product(product: RadarProduct) -> Self {
        let p = product.base_product();
        match p {
            RadarProduct::Reflectivity | RadarProduct::CompositeReflectivity => Self::from_entries(vec![
                ce(-30.0,0,0,0,0), ce(-20.0,100,100,100,180), ce(-10.0,150,150,150,200),
                ce(0.0,118,118,118,220), ce(5.0,0,236,236,255), ce(10.0,1,160,246,255),
                ce(15.0,0,0,246,255), ce(20.0,0,255,0,255), ce(25.0,0,200,0,255),
                ce(30.0,0,144,0,255), ce(35.0,255,255,0,255), ce(40.0,231,192,0,255),
                ce(45.0,255,144,0,255), ce(50.0,255,0,0,255), ce(55.0,214,0,0,255),
                ce(60.0,192,0,0,255), ce(65.0,255,0,255,255), ce(70.0,153,85,201,255),
                ce(75.0,255,255,255,255),
            ]),
            RadarProduct::Velocity | RadarProduct::StormRelativeVelocity => Self::from_entries(vec![
                ce(-120.0,255,0,255,255), ce(-100.0,200,0,200,255), ce(-80.0,128,0,0,255),
                ce(-64.0,255,0,0,255), ce(-50.0,192,0,0,255), ce(-36.0,255,127,0,255),
                ce(-26.0,255,200,0,255), ce(-20.0,255,230,137,255), ce(-10.0,141,0,0,255),
                ce(-1.0,100,55,55,200), ce(0.0,0,0,0,0), ce(1.0,55,100,55,200),
                ce(10.0,0,141,0,255), ce(20.0,137,230,137,255), ce(26.0,0,200,0,255),
                ce(36.0,0,255,127,255), ce(50.0,0,192,0,255), ce(64.0,0,0,255,255),
                ce(80.0,0,0,128,255), ce(100.0,0,200,200,255), ce(120.0,0,255,255,255),
            ]),
            RadarProduct::SpectrumWidth => Self::from_entries(vec![
                ce(0.0,0,0,0,0), ce(2.0,100,100,100,200), ce(5.0,0,150,0,255),
                ce(10.0,0,255,0,255), ce(15.0,255,255,0,255), ce(20.0,255,150,0,255),
                ce(25.0,255,0,0,255), ce(30.0,200,0,0,255), ce(40.0,255,255,255,255),
            ]),
            RadarProduct::DifferentialReflectivity => Self::from_entries(vec![
                ce(-8.0,0,0,128,255), ce(-4.0,0,0,255,255), ce(-2.0,0,150,255,255),
                ce(-1.0,0,200,200,255), ce(0.0,100,100,100,200), ce(1.0,0,200,0,255),
                ce(2.0,255,255,0,255), ce(4.0,255,128,0,255), ce(6.0,255,0,0,255),
                ce(8.0,200,0,200,255),
            ]),
            RadarProduct::CorrelationCoefficient => Self::from_entries(vec![
                ce(0.2,0,0,0,0), ce(0.5,128,0,128,255), ce(0.7,0,0,200,255),
                ce(0.8,0,150,255,255), ce(0.85,0,200,200,255), ce(0.90,0,200,0,255),
                ce(0.93,255,255,0,255), ce(0.95,255,128,0,255), ce(0.97,255,0,0,255),
                ce(0.99,200,0,200,255), ce(1.05,255,255,255,255),
            ]),
            RadarProduct::SpecificDifferentialPhase => Self::from_entries(vec![
                ce(-2.0,128,0,128,255), ce(-1.0,0,0,200,255), ce(0.0,100,100,100,200),
                ce(0.5,0,200,0,255), ce(1.0,0,255,0,255), ce(2.0,255,255,0,255),
                ce(3.0,255,200,0,255), ce(5.0,255,128,0,255), ce(7.0,255,0,0,255),
                ce(10.0,200,0,200,255),
            ]),
            _ => Self::from_entries(vec![
                ce(-30.0,0,0,0,0), ce(0.0,0,0,246,255), ce(20.0,0,255,0,255),
                ce(40.0,255,255,0,255), ce(60.0,255,0,0,255), ce(80.0,255,255,255,255),
            ]),
        }
    }
}
