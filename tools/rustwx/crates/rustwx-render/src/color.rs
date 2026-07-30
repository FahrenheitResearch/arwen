/// RGBA color (8-bit per channel).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Rgba {
    pub const TRANSPARENT: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };
    pub const WHITE: Self = Self {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };
    pub const BLACK: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };

    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    pub const fn with_alpha(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Parse a "#RRGGBB" hex string.
    pub fn from_hex(hex: &str) -> Self {
        let h = hex.trim_start_matches('#');
        let r = u8::from_str_radix(&h[0..2], 16).unwrap_or(0);
        let g = u8::from_str_radix(&h[2..4], 16).unwrap_or(0);
        let b = u8::from_str_radix(&h[4..6], 16).unwrap_or(0);
        Self { r, g, b, a: 255 }
    }

    /// Convert to image::Rgba for the `image` crate.
    pub fn to_image_rgba(self) -> image::Rgba<u8> {
        image::Rgba([self.r, self.g, self.b, self.a])
    }
}

// ---------------------------------------------------------------------------
// Colour interpolation helpers used by the weather palette anchor tables.
// ---------------------------------------------------------------------------

/// Linearly interpolate `n` colours across a list of anchor hex strings.
pub fn lerp_hex(anchors: &[&str], n: usize) -> Vec<Rgba> {
    let rgb: Vec<Rgba> = anchors.iter().map(|h| Rgba::from_hex(h)).collect();
    lerp_rgba(&rgb, n)
}

/// Linearly interpolate `n` colours across a list of Rgba anchors.
pub fn lerp_rgba(anchors: &[Rgba], n: usize) -> Vec<Rgba> {
    if n == 0 {
        return vec![];
    }
    if n == 1 || anchors.len() <= 1 {
        return vec![anchors[0]; n.max(1)];
    }
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64 * (anchors.len() - 1) as f64;
        let lo = t as usize;
        let hi = (lo + 1).min(anchors.len() - 1);
        let f = t - lo as f64;
        let a = &anchors[lo];
        let b = &anchors[hi];
        result.push(Rgba::with_alpha(
            (a.r as f64 + f * (b.r as f64 - a.r as f64)).round() as u8,
            (a.g as f64 + f * (b.g as f64 - a.g as f64)).round() as u8,
            (a.b as f64 + f * (b.b as f64 - a.b as f64)).round() as u8,
            (a.a as f64 + f * (b.a as f64 - a.a as f64)).round() as u8,
        ));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_interpolation_preserves_alpha() {
        let colors = lerp_rgba(
            &[
                Rgba::with_alpha(0, 0, 0, 0),
                Rgba::with_alpha(10, 20, 30, 200),
            ],
            3,
        );

        assert_eq!(colors[0].a, 0);
        assert_eq!(colors[1].a, 100);
        assert_eq!(colors[2].a, 200);
    }
}
