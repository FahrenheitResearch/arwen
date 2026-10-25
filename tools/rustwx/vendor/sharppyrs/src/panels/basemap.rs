//! World basemap for the sounding locator insets (the Location panel and the
//! hodograph's corner inset): coastlines, country borders, large lakes and
//! the US/Canada/Mexico first-order admin lines, from the embedded Natural
//! Earth data in [`super::world_map_data`].
//!
//! Breakage this fixes (hex production check 2026-10-24): the insets drew a
//! CONUS-only basemap (state lines + coast cropped to lon -125.5..-66,
//! lat 24..50.5), so every sounding outside that box -- the tropics, Europe,
//! Asia, any hex global point -- got an empty inset with just the crosshair.

use egui::{Painter, Pos2, Shape, Stroke};

use super::world_map_data as data;

struct Layer {
    starts: &'static [u32],
    bbox: &'static [i16],
    xy: &'static [i16],
}

const LAYERS: [Layer; 4] = [
    Layer {
        starts: &data::COAST_STARTS,
        bbox: &data::COAST_BBOX,
        xy: &data::COAST_XY,
    },
    Layer {
        starts: &data::LAKES_STARTS,
        bbox: &data::LAKES_BBOX,
        xy: &data::LAKES_XY,
    },
    Layer {
        starts: &data::BORDERS_STARTS,
        bbox: &data::BORDERS_BBOX,
        xy: &data::BORDERS_XY,
    },
    Layer {
        starts: &data::STATES_STARTS,
        bbox: &data::STATES_BBOX,
        xy: &data::STATES_XY,
    },
];

/// A lon/lat window in degrees. `west`/`east` may run past +-180 near the
/// dateline; the data is drawn shifted by 360 degrees to fill it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Extent {
    pub west: f64,
    pub south: f64,
    pub east: f64,
    pub north: f64,
}

impl Extent {
    fn scaled_about(self, lon: f64, lat: f64, factor: f64) -> Self {
        Self {
            west: lon - (lon - self.west) * factor,
            south: lat - (lat - self.south) * factor,
            east: lon + (self.east - lon) * factor,
            north: lat + (self.north - lat) * factor,
        }
    }

    fn shrunk(self, fraction: f64) -> Self {
        let dx = (self.east - self.west) * fraction;
        let dy = (self.north - self.south) * fraction;
        Self {
            west: self.west + dx,
            south: self.south + dy,
            east: self.east - dx,
            north: self.north - dy,
        }
    }
}

/// The 360-degree shifts under which the -180..180 data overlaps `ext`.
fn shifts(ext: Extent) -> impl Iterator<Item = f64> {
    [-360.0, 0.0, 360.0]
        .into_iter()
        .filter(move |s| ext.west < 180.0 + s && ext.east > -180.0 + s)
}

/// Call `f(points, shift)` for every polyline whose bbox meets `ext`
/// (points in hundredths of a degree, `shift` in degrees).
fn for_each_visible(ext: Extent, mut f: impl FnMut(&[i16], f64)) {
    for shift in shifts(ext) {
        let w = ((ext.west - shift) * 100.0).floor();
        let e = ((ext.east - shift) * 100.0).ceil();
        let s = (ext.south * 100.0).floor();
        let n = (ext.north * 100.0).ceil();
        for layer in &LAYERS {
            for i in 0..layer.starts.len() - 1 {
                let b = &layer.bbox[4 * i..4 * i + 4];
                if f64::from(b[2]) < w || f64::from(b[0]) > e || f64::from(b[3]) < s || f64::from(b[1]) > n {
                    continue;
                }
                let (a, z) = (layer.starts[i] as usize, layer.starts[i + 1] as usize);
                f(&layer.xy[2 * a..2 * z], shift);
            }
        }
    }
}

/// Does the segment p0-p1 cross or touch the box (Liang-Barsky)?
fn segment_meets(p0: (f64, f64), p1: (f64, f64), ext: Extent) -> bool {
    let (dx, dy) = (p1.0 - p0.0, p1.1 - p0.1);
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for (p, q) in [
        (-dx, p0.0 - ext.west),
        (dx, ext.east - p0.0),
        (-dy, p0.1 - ext.south),
        (dy, ext.north - p0.1),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return false;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
            if t0 > t1 {
                return false;
            }
        }
    }
    true
}

/// Is any basemap line inside `ext`?
pub(crate) fn has_outline_in(ext: Extent) -> bool {
    let mut found = false;
    for_each_visible(ext, |xy, shift| {
        if found {
            return;
        }
        let pt = |k: usize| (f64::from(xy[2 * k]) / 100.0 + shift, f64::from(xy[2 * k + 1]) / 100.0);
        found = (1..xy.len() / 2).any(|k| segment_meets(pt(k - 1), pt(k), ext));
    });
    found
}

/// Widen `ext` about the sounding point (doubling, up to 8x) until some
/// coastline or border falls inside its central 80%, so a point over open
/// ocean still shows where it is instead of an empty frame. A view that
/// already holds an outline is returned unchanged.
pub(crate) fn widen_until_outline(ext: Extent, lon: f64, lat: f64) -> Extent {
    for factor in [1.0, 2.0, 4.0] {
        let candidate = ext.scaled_about(lon, lat, factor);
        if has_outline_in(candidate.shrunk(0.1)) {
            return candidate;
        }
    }
    ext.scaled_about(lon, lat, 8.0)
}

/// Draw every basemap line in `ext` through `map_point(lon, lat)`.
pub(crate) fn draw(painter: &Painter, ext: Extent, map_point: impl Fn(f64, f64) -> Pos2, stroke: Stroke) {
    for_each_visible(ext, |xy, shift| {
        let pts: Vec<Pos2> = xy
            .chunks_exact(2)
            .map(|p| map_point(f64::from(p[0]) / 100.0 + shift, f64::from(p[1]) / 100.0))
            .collect();
        if pts.len() >= 2 {
            painter.add(Shape::line(pts, stroke));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn around(lon: f64, lat: f64, half_lon: f64, half_lat: f64) -> Extent {
        Extent {
            west: lon - half_lon,
            south: lat - half_lat,
            east: lon + half_lon,
            north: lat + half_lat,
        }
    }

    #[test]
    fn the_basemap_covers_points_outside_conus() {
        // Production check 2026-10-24: Paris, Tokyo, Sydney and Nairobi
        // views drew nothing with the CONUS-only basemap.
        for (lon, lat) in [(2.3, 48.8), (139.7, 35.7), (151.2, -33.9), (36.8, -1.3)] {
            assert!(has_outline_in(around(lon, lat, 6.0, 4.0)), "{lat} {lon}");
        }
        // The CONUS state lines are still there: a window across the
        // Kansas-Nebraska line (40N) holds no coast or border, only it.
        assert!(has_outline_in(around(-99.0, 40.0, 1.5, 0.5)));
    }

    #[test]
    fn an_open_ocean_point_widens_until_a_coast_is_in_view() {
        // 12N 102.3W, the East Pacific sounding of the production check.
        let ext = around(-102.3, 12.0, 6.5, 4.0);
        let wide = widen_until_outline(ext, -102.3, 12.0);
        assert!(wide.north - wide.south > ext.north - ext.south);
        assert!(has_outline_in(wide.shrunk(0.1)));
        // A view that already has outlines is left alone.
        let paris = around(2.3, 48.8, 6.0, 4.0);
        assert_eq!(widen_until_outline(paris, 2.3, 48.8), paris);
    }

    #[test]
    fn a_view_across_the_dateline_finds_data_on_both_sides() {
        // Chukotka and western Alaska sit at -175..-165, i.e. 185..195
        // when a view centred near 180 runs past the dateline.
        assert!(has_outline_in(Extent { west: 186.0, south: 62.0, east: 194.0, north: 68.0 }));
        assert!(has_outline_in(around(178.0, 65.0, 10.0, 4.0)));
    }
}
