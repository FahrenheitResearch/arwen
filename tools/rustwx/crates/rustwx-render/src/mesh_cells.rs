//! The polygon-mesh layer: one filled polygon per model cell, with a
//! hairline along every cell edge.
//!
//! WHAT BREAKAGE THIS PREVENTS (gate law): an unstructured mesh
//! drawn as a raster lies about where its numbers are.  Every other product
//! family in this crate starts from a structured `(ny, nx)` field and
//! rasterises it; putting a Voronoi mesh through that door needs a regrid
//! onto a lat/lon frame first, and the regrid invents values between cell
//! centres, hides the mesh's own refinement, and smears a single-cell
//! response over its neighbours.  A response that lives in forty cells is
//! exactly the thing a pair difference is drawn to show.
//!
//! So this layer takes the cells themselves: a ring of projected vertices
//! and one value per cell.  A cell with no value -- a regional mesh's outer
//! relaxation ring, a masked column -- takes the theme's empty fill, which is
//! what keeps the mesh visible where the field is not.
//!
//! The two passes are deliberate.  Every cell is filled first and every edge
//! stroked afterwards, so a shared edge is drawn on top of both its cells
//! rather than half-covered by whichever neighbour filled last.

use crate::color::Rgba;
use crate::draw;
use image::RgbaImage;
use serde::{Deserialize, Serialize};

/// One mesh cell: its boundary ring in PROJECTED map coordinates (the same
/// space [`crate::ProjectedDomain`] is in) and the value that colours it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshCell {
    pub ring: Vec<(f64, f64)>,
    /// `None` draws the theme's empty fill instead of a colormap colour.
    #[serde(default)]
    pub value: Option<f64>,
}

/// Every cell of one mesh, in draw order.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MeshCellsLayer {
    pub cells: Vec<MeshCell>,
}

impl MeshCellsLayer {
    pub fn new(cells: Vec<MeshCell>) -> Self {
        Self { cells }
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}

/// The resolved linework and empty-cell fill for one mesh draw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshDrawStyle {
    pub edge: Rgba,
    /// Hairline width in the pixels of the surface being drawn on, so a
    /// supersampled pass scales it before calling.
    pub edge_width_px: f32,
    pub draw_edges: bool,
    pub empty: Rgba,
}

impl MeshDrawStyle {
    pub fn from_theme(theme: crate::theme::MeshTheme, supersample: u32) -> Self {
        Self {
            edge: theme.edge,
            edge_width_px: theme.edge_width_px() * supersample.max(1) as f32,
            draw_edges: theme.draw_edges,
            empty: theme.empty,
        }
    }
}

/// Fill every ring, then stroke every ring.
///
/// `rings` and `colors` are parallel and in PIXEL coordinates of `img`.
/// `clip` is the inclusive pixel rectangle the mesh may touch (the map
/// frame); without it a cell whose projection falls outside the frame would
/// paint over the title row and the colorbar.
pub fn draw_mesh_cells_pixels(
    img: &mut RgbaImage,
    rings: &[Vec<(f64, f64)>],
    colors: &[Rgba],
    style: MeshDrawStyle,
    clip: Option<(i32, i32, i32, i32)>,
) {
    debug_assert_eq!(rings.len(), colors.len());
    // fill_polygon takes a ring LIST (outer ring plus holes); a mesh cell is
    // one ring, so the vector is reused rather than allocated per cell.
    let mut one_ring: Vec<Vec<(f64, f64)>> = vec![Vec::new()];
    for (ring, color) in rings.iter().zip(colors.iter()) {
        if ring.len() < 3 || color.a == 0 {
            continue;
        }
        one_ring[0].clear();
        one_ring[0].extend_from_slice(ring);
        draw::fill_polygon(img, &one_ring, *color, clip);
    }

    if !style.draw_edges || style.edge.a == 0 || style.edge_width_px <= 0.0 {
        return;
    }
    // A cell smaller than its own outline is not outlined.  On a global mesh
    // framed at continental scale the fine cells are about a pixel across,
    // and a hairline drawn round each of them covers the fill entirely: the
    // refined region came out DARKER than its neighbours and the panel
    // reported the mesh where the field belonged.  The measurement has to be
    // the field; the mesh is drawn where there is room to draw it.
    let floor = f64::from(style.edge_width_px) * 2.0;
    for ring in rings {
        if ring.len() < 2 {
            continue;
        }
        if ring_extent(ring) < floor {
            continue;
        }
        for index in 0..ring.len() {
            let (x0, y0) = ring[index];
            let (x1, y1) = ring[(index + 1) % ring.len()];
            stroke_segment(img, x0, y0, x1, y1, style.edge, style.edge_width_px, clip);
        }
    }
}

/// The larger side of a ring's pixel bounding box.
fn ring_extent(ring: &[(f64, f64)]) -> f64 {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &(x, y) in ring {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    if !min_x.is_finite() || !min_y.is_finite() {
        return 0.0;
    }
    (max_x - min_x).max(max_y - min_y)
}

/// One anti-aliased segment, clipped to `clip`.
///
/// `draw::draw_line_aa_width` takes an integer width and clips only to the
/// image, and a mesh hairline is both fractional (a supersampled 1 px edge is
/// 2 px on the hi-res canvas but a half-pixel edge is a legitimate ask) and
/// has to stop at the map frame.  Same coverage rule as the shared kernel:
/// distance to the segment, one pixel of feather.
fn stroke_segment(
    img: &mut RgbaImage,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    color: Rgba,
    width_px: f32,
    clip: Option<(i32, i32, i32, i32)>,
) {
    if !x0.is_finite() || !y0.is_finite() || !x1.is_finite() || !y1.is_finite() {
        return;
    }
    let half = f64::from(width_px).max(0.05) * 0.5;
    let radius = half + 1.0;
    let img_w = img.width() as i32;
    let img_h = img.height() as i32;
    let (cx0, cy0, cx1, cy1) = match clip {
        Some((a, b, c, d)) => (a.max(0), b.max(0), c.min(img_w - 1), d.min(img_h - 1)),
        None => (0, 0, img_w - 1, img_h - 1),
    };
    if cx1 < cx0 || cy1 < cy0 {
        return;
    }
    let min_x = ((x0.min(x1) - radius).floor() as i32).max(cx0);
    let max_x = ((x0.max(x1) + radius).ceil() as i32).min(cx1);
    let min_y = ((y0.min(y1) - radius).floor() as i32).max(cy0);
    let max_y = ((y0.max(y1) + radius).ceil() as i32).min(cy1);
    if max_x < min_x || max_y < min_y {
        return;
    }
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px = f64::from(x) + 0.5;
            let py = f64::from(y) + 0.5;
            let distance = distance_to_segment(px, py, x0, y0, x1, y1);
            let coverage = (half + 0.5 - distance).clamp(0.0, 1.0);
            if coverage <= 0.0 {
                continue;
            }
            let alpha = ((f64::from(color.a) * coverage).round()).clamp(0.0, 255.0) as u8;
            draw::blend_pixel(img, x, y, Rgba::with_alpha(color.r, color.g, color.b, alpha));
        }
    }
}

fn distance_to_segment(px: f64, py: f64, x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len_sq = dx * dx + dy * dy;
    if len_sq <= 1e-12 {
        let ox = px - x0;
        let oy = py - y0;
        return (ox * ox + oy * oy).sqrt();
    }
    let t = (((px - x0) * dx + (py - y0) * dy) / len_sq).clamp(0.0, 1.0);
    let proj_x = x0 + t * dx;
    let proj_y = y0 + t * dy;
    let ox = px - proj_x;
    let oy = py - proj_y;
    (ox * ox + oy * oy).sqrt()
}

/// A regular hexagon of circumradius `r` centred on `(cx, cy)`, flat-top.
///
/// Test scaffolding kept beside the rasteriser it exercises, and used by the
/// mesh family's own fixtures.
pub fn regular_hexagon(cx: f64, cy: f64, r: f64) -> Vec<(f64, f64)> {
    (0..6)
        .map(|k| {
            let angle = std::f64::consts::PI / 3.0 * f64::from(k);
            (cx + r * angle.cos(), cy + r * angle.sin())
        })
        .collect()
}

/// The seven cells of a planted hex patch: one centre and its six
/// neighbours, spaced so the tiling closes.
pub fn planted_hex_patch(cx: f64, cy: f64, r: f64) -> Vec<Vec<(f64, f64)>> {
    let step = r * 3.0_f64.sqrt();
    let mut patch = vec![regular_hexagon(cx, cy, r)];
    for k in 0..6 {
        let angle = std::f64::consts::PI / 3.0 * f64::from(k) + std::f64::consts::FRAC_PI_6;
        patch.push(regular_hexagon(
            cx + step * angle.cos(),
            cy + step * angle.sin(),
            r,
        ));
    }
    patch
}

#[cfg(test)]
mod tests {
    use super::*;

    fn style() -> MeshDrawStyle {
        MeshDrawStyle {
            edge: Rgba::with_alpha(0x22, 0x31, 0x37, 255),
            edge_width_px: 1.0,
            draw_edges: true,
            empty: Rgba::with_alpha(0x12, 0x19, 0x1c, 255),
        }
    }

    #[test]
    fn a_planted_seven_cell_patch_fills_each_cell_with_its_own_colour() {
        let rings = planted_hex_patch(120.0, 120.0, 28.0);
        assert_eq!(rings.len(), 7);
        let colors = vec![
            Rgba::with_alpha(255, 0, 0, 255),
            Rgba::with_alpha(0, 255, 0, 255),
            Rgba::with_alpha(0, 0, 255, 255),
            Rgba::with_alpha(255, 255, 0, 255),
            Rgba::with_alpha(255, 0, 255, 255),
            Rgba::with_alpha(0, 255, 255, 255),
            Rgba::with_alpha(200, 200, 200, 255),
        ];
        let mut img = RgbaImage::from_pixel(240, 240, image::Rgba([0, 0, 0, 255]));
        draw_mesh_cells_pixels(&mut img, &rings, &colors, style(), None);

        // The centre pixel of every cell carries that cell's fill, so the
        // rasteriser put each value where its own cell is and nowhere else.
        let step = 28.0 * 3.0_f64.sqrt();
        let mut centres = vec![(120.0, 120.0)];
        for k in 0..6 {
            let angle = std::f64::consts::PI / 3.0 * f64::from(k) + std::f64::consts::FRAC_PI_6;
            centres.push((120.0 + step * angle.cos(), 120.0 + step * angle.sin()));
        }
        for (index, (cx, cy)) in centres.iter().enumerate() {
            let pixel = img.get_pixel(cx.round() as u32, cy.round() as u32).0;
            let want = colors[index];
            assert_eq!(
                [pixel[0], pixel[1], pixel[2]],
                [want.r, want.g, want.b],
                "cell {index} centre"
            );
        }

        // A point well outside the patch is untouched canvas.
        assert_eq!(img.get_pixel(4, 4).0, [0, 0, 0, 255]);
    }

    #[test]
    fn a_valueless_cell_takes_the_empty_fill_and_the_edges_are_drawn_on_top() {
        let rings = vec![regular_hexagon(60.0, 60.0, 30.0)];
        let mut img = RgbaImage::from_pixel(120, 120, image::Rgba([0, 0, 0, 255]));
        draw_mesh_cells_pixels(&mut img, &rings, &[style().empty], style(), None);
        assert_eq!(
            img.get_pixel(60, 60).0,
            [0x12, 0x19, 0x1c, 255],
            "the cell interior is the empty fill"
        );
        // The right vertex of a flat-top hexagon sits at (90, 60); the edge
        // hairline runs through the pixels just inside it.
        let edge_pixel = img.get_pixel(89, 60).0;
        assert_ne!(
            [edge_pixel[0], edge_pixel[1], edge_pixel[2]],
            [0x12, 0x19, 0x1c],
            "the hairline is drawn over the fill at the cell boundary"
        );
    }

    #[test]
    fn a_cell_smaller_than_its_own_hairline_keeps_its_fill() {
        // Two cells, one of them sub-pixel: the small one must still report
        // its value, not the outline that would swallow it.
        let big = regular_hexagon(30.0, 30.0, 20.0);
        let tiny = regular_hexagon(80.0, 30.0, 0.6);
        let colors = vec![
            Rgba::with_alpha(10, 200, 10, 255),
            Rgba::with_alpha(220, 40, 40, 255),
        ];
        let mut img = RgbaImage::from_pixel(120, 60, image::Rgba([0, 0, 0, 255]));
        draw_mesh_cells_pixels(&mut img, &[big, tiny], &colors, style(), None);
        let pixel = img.get_pixel(80, 30).0;
        assert_eq!(
            [pixel[0], pixel[1], pixel[2]],
            [220, 40, 40],
            "the sub-pixel cell reports its own value"
        );
        // The cell with room for an outline still gets one.
        let edge = img.get_pixel(49, 30).0;
        assert_ne!([edge[0], edge[1], edge[2]], [10, 200, 10]);
    }

    #[test]
    fn the_clip_rectangle_keeps_a_cell_out_of_the_chrome() {
        let rings = vec![regular_hexagon(60.0, 60.0, 40.0)];
        let colors = vec![Rgba::with_alpha(255, 0, 0, 255)];
        let mut img = RgbaImage::from_pixel(120, 120, image::Rgba([0, 0, 0, 255]));
        draw_mesh_cells_pixels(&mut img, &rings, &colors, style(), Some((0, 0, 119, 59)));
        assert_eq!(
            img.get_pixel(60, 50).0,
            [255, 0, 0, 255],
            "inside the clip the cell is filled"
        );
        assert_eq!(
            img.get_pixel(60, 70).0,
            [0, 0, 0, 255],
            "below the clip nothing was drawn"
        );
    }

    #[test]
    fn edges_can_be_turned_off_and_the_fill_survives() {
        let rings = vec![regular_hexagon(60.0, 60.0, 30.0)];
        let colors = vec![Rgba::with_alpha(10, 120, 200, 255)];
        let mut off = style();
        off.draw_edges = false;
        let mut img = RgbaImage::from_pixel(120, 120, image::Rgba([0, 0, 0, 255]));
        draw_mesh_cells_pixels(&mut img, &rings, &colors, off, None);
        assert_eq!(img.get_pixel(60, 60).0, [10, 120, 200, 255]);
        assert_eq!(
            img.get_pixel(89, 60).0,
            [10, 120, 200, 255],
            "with edges off the boundary pixel is still the fill"
        );
    }
}
