//! Aspect-aware canvas planning.
//!
//! The canvas is sized from the map, and the map from the grid's own
//! projected aspect.  The old order ran the other way: every panel was
//! drawn on a fixed 1200x900 canvas, the projected extent was padded out
//! to that canvas's map ratio, and the padding was painted canvas colour.
//! A square grid came out with a white band left and right (45 to 55% of
//! the canvas on the survey's square nests) and a wide grid letterboxed
//! top and bottom (66%).  Three pixel-scan passes then tried to clean
//! that up, chosen by where the domain sat on the Earth.
//!
//! Here the map gets a fixed pixel budget at the grid's aspect, clamped
//! to a band, and the chrome (header, colour bar, margins) is added
//! around it.  Every size is a row in `layouts/default.json`: the rules
//! are data, the solver is arithmetic.  A canvas depends only on the
//! domain, the size class and the table, never on the product, so every
//! product of one domain shares one size and a loop does not jitter.

use serde::{Deserialize, Serialize};
use std::sync::{OnceLock, RwLock};

const DEFAULT_TABLE_JSON: &str = include_str!("../layouts/default.json");

/// The layout table (`layouts/default.json`), at size-class scale 1.0.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct LayoutTable {
    pub map_area_px: f64,
    pub map_w_max: u32,
    pub map_h_max: u32,
    pub map_short_min: u32,
    pub aspect_min: f64,
    pub aspect_max: f64,
    pub margin: u32,
    pub canvas_min_w: u32,
    pub header: HeaderTable,
    pub bar: BarTable,
    pub map_frame: MapFrameTable,
    pub barb_spacing_px: f64,
    /// A masked field whose grid cell spans this many map pixels or more
    /// draws its cells sharp (nearest sampling).  0 turns the rule off,
    /// which is the default: on by default it drew echo, rain and CAPE as
    /// blocks beside smooth temperature and wind on the same domain, and
    /// the 1 km seeding frames came out looking low-res.
    pub sharp_cell_px: f64,
    pub size_classes: SizeClassTable,
    pub sheet: SheetTable,
    pub section: SectionTable,
    pub layers: LayersTable,
}

/// Which map layers a frame carries, keyed on the frame's short side in
/// km: a county mesh at continental scale is clutter, a metro frame with
/// no place names has no reference.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct LayersTable {
    pub counties_max_short_km: f64,
    pub places: Vec<PlaceTierRow>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct PlaceTierRow {
    /// The largest short side (km) this row covers; `null` is every size.
    pub max_short_km: Option<f64>,
    pub tier: String,
    /// At most this many places before the tier's own multiplier, and
    /// no two closer than `min_spacing_km`.
    pub max_count: u32,
    pub min_spacing_km: u32,
}

impl LayersTable {
    pub fn counties_visible(&self, short_km: f64) -> bool {
        !short_km.is_finite() || short_km <= self.counties_max_short_km
    }

    /// The place density a frame of `short_km` carries.
    pub fn place_tier(&self, short_km: f64) -> &str {
        self.place_row(short_km).map(|row| row.tier.as_str()).unwrap_or("none")
    }

    pub fn place_row(&self, short_km: f64) -> Option<&PlaceTierRow> {
        self.places
            .iter()
            .find(|row| row.max_short_km.is_none_or(|max| short_km <= max))
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct HeaderTable {
    pub pad_top: u32,
    pub title_row_h: u32,
    pub meta_row_h: u32,
    pub pad_bottom: u32,
    /// Below this content width the valid time moves off the title row
    /// and the header takes three rows.
    pub two_row_min_content_w: u32,
    pub title_px: f32,
    /// A title too long for its row steps down to this size before it is
    /// ever cut.
    pub title_min_px: f32,
    pub meta_px: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct BarTable {
    /// At or above this map aspect the bar goes under the map at full map
    /// width; below it, on the right at exactly map height.
    pub bottom_min_aspect: f64,
    pub right: RightBarTable,
    pub bottom: BottomBarTable,
    pub tick_px: f32,
    pub tick_spacing_vertical_px: u32,
    pub tick_spacing_horizontal_px: u32,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct RightBarTable {
    pub gap: u32,
    pub thickness: u32,
    pub tick_gap: u32,
    pub label_w: u32,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct BottomBarTable {
    pub gap: u32,
    pub thickness: u32,
    pub tick_gap: u32,
    pub label_h: u32,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct MapFrameTable {
    pub outline_px: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct SizeClassRow {
    pub scale: f64,
    pub font: f64,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct SizeClassTable {
    pub standard: SizeClassRow,
    pub phone: SizeClassRow,
    pub large: SizeClassRow,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct SheetTable {
    pub map_area_px: f64,
    pub width_max: u32,
    pub gutter: u32,
    pub label_h: u32,
    pub label_px: f32,
    /// A pair sheet's run labels: larger than a member label, because the
    /// run is the comparison's subject.
    pub leg_label_px: f32,
    pub target_aspect: f64,
    pub empty_cell_weight: f64,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct SectionTable {
    pub plot_aspect: f64,
    pub plot_area_px: f64,
    pub axis_left: u32,
    pub axis_bottom: u32,
}

/// The size class a frame is drawn for.  Live frames always draw
/// `Standard`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SizeClass {
    #[default]
    Standard,
    Phone,
    Large,
}

impl SizeClass {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "standard" => Some(Self::Standard),
            "phone" => Some(Self::Phone),
            "large" => Some(Self::Large),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Phone => "phone",
            Self::Large => "large",
        }
    }
}

/// Which side of the map the colour bar is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BarSide {
    Right,
    Bottom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PlanRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl PlanRect {
    pub fn right(self) -> u32 {
        self.x + self.w
    }

    pub fn bottom(self) -> u32 {
        self.y + self.h
    }

    fn scaled(self, factor: u32) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
            w: self.w * factor,
            h: self.h * factor,
        }
    }
}

/// One frame's geometry: where the map, the header and the colour bar go
/// on a canvas sized for them.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CanvasPlan {
    pub canvas_w: u32,
    pub canvas_h: u32,
    pub map: PlanRect,
    /// The header band: the full content width (canvas less its margins),
    /// never the map's width, so a narrow map cannot cut the header.
    /// Zero height for a sheet member, whose labels belong to the sheet.
    pub header: PlanRect,
    pub header_rows: u8,
    pub pad_top: u32,
    pub title_row_h: u32,
    pub meta_row_h: u32,
    pub bar: Option<PlanRect>,
    pub bar_side: BarSide,
    /// The room past the bar for its tick labels: a width beside a right
    /// bar, a height under a bottom one.
    pub bar_label_span: u32,
    pub bar_tick_gap: u32,
    pub title_px: f32,
    pub title_min_px: f32,
    pub meta_px: f32,
    pub tick_px: f32,
    pub tick_spacing_px: u32,
    /// The grid's own aspect (width over height, projected), unclamped.
    pub grid_aspect: f64,
    /// `map.w / map.h`.
    pub map_aspect: f64,
    /// True when the grid's aspect was outside the table's band, so the
    /// map carries basemap past the grid on two sides.
    pub aspect_clamped: bool,
    pub scale: f64,
    pub barb_spacing_px: f64,
    pub outline_px: f32,
    pub sharp_cell_px: f64,
}

impl CanvasPlan {
    /// This plan at an integer supersample factor.
    pub fn scaled(&self, factor: u32) -> Self {
        if factor <= 1 {
            return *self;
        }
        let f = factor as f32;
        Self {
            canvas_w: self.canvas_w * factor,
            canvas_h: self.canvas_h * factor,
            map: self.map.scaled(factor),
            header: self.header.scaled(factor),
            pad_top: self.pad_top * factor,
            title_row_h: self.title_row_h * factor,
            meta_row_h: self.meta_row_h * factor,
            bar: self.bar.map(|bar| bar.scaled(factor)),
            bar_label_span: self.bar_label_span * factor,
            bar_tick_gap: self.bar_tick_gap * factor,
            title_px: self.title_px * f,
            title_min_px: self.title_min_px * f,
            meta_px: self.meta_px * f,
            tick_px: self.tick_px * f,
            tick_spacing_px: self.tick_spacing_px * factor,
            scale: self.scale * factor as f64,
            barb_spacing_px: self.barb_spacing_px * factor as f64,
            outline_px: self.outline_px * f,
            ..*self
        }
    }

    /// The same frame with no colour bar: a product whose fill carries no
    /// quantity keeps its domain's canvas, and its map stays where every
    /// other product of the domain has it.
    pub fn without_bar(&self) -> Self {
        Self { bar: None, ..*self }
    }
}

/// A multi-panel sheet: one header, a grid of maps with a label strip
/// over each, and one shared colour bar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SheetPlan {
    pub canvas_w: u32,
    pub canvas_h: u32,
    pub rows: u32,
    pub columns: u32,
    pub header: PlanRect,
    pub header_rows: u8,
    /// Each cell's map rectangle, row-major.
    pub cells: Vec<PlanRect>,
    /// Each cell's label strip, directly above its map.
    pub labels: Vec<PlanRect>,
    pub label_px: f32,
    pub bar: Option<PlanRect>,
    pub bar_side: BarSide,
    /// The plan each member is rendered with: a canvas exactly one cell's
    /// map, no header, no bar.
    pub member: CanvasPlan,
    /// The same header geometry a single map uses, so a sheet's title,
    /// valid time and metadata sit exactly where a map's do.
    pub frame: CanvasPlan,
}

/// A section's canvas: the map header over a plot area of fixed aspect.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SectionPlan {
    pub canvas_w: u32,
    pub canvas_h: u32,
    pub plot: PlanRect,
}

fn even(value: f64) -> u32 {
    ((value / 2.0).round() * 2.0).max(2.0) as u32
}

fn px(value: u32, scale: f64) -> u32 {
    (value as f64 * scale).round() as u32
}

impl LayoutTable {
    /// The compiled-in table.  A unit test parses it, so a malformed row
    /// fails the build's tests, never a render.
    pub fn builtin() -> &'static LayoutTable {
        static TABLE: OnceLock<LayoutTable> = OnceLock::new();
        TABLE.get_or_init(|| {
            serde_json::from_str(DEFAULT_TABLE_JSON)
                .expect("layouts/default.json is compiled in and parsed by a unit test")
        })
    }

    pub fn size_class(&self, class: SizeClass) -> &SizeClassRow {
        match class {
            SizeClass::Standard => &self.size_classes.standard,
            SizeClass::Phone => &self.size_classes.phone,
            SizeClass::Large => &self.size_classes.large,
        }
    }

    /// Clamp a grid aspect into the table's band.  A degenerate aspect
    /// (zero, negative, not finite) plans as a square.
    pub fn clamp_aspect(&self, aspect: f64) -> f64 {
        if aspect.is_finite() && aspect > 0.0 {
            aspect.clamp(self.aspect_min, self.aspect_max)
        } else {
            1.0
        }
    }

    /// Map pixels for an aspect and a pixel-area budget, inside the caps
    /// and above the short-side floor, both sides even.
    fn map_size(&self, aspect: f64, area: f64, scale: f64) -> (u32, u32) {
        let mut w = (area * aspect).sqrt();
        let mut h = w / aspect;
        let w_max = self.map_w_max as f64 * scale;
        let h_max = self.map_h_max as f64 * scale;
        if w > w_max {
            w = w_max;
            h = w / aspect;
        }
        if h > h_max {
            h = h_max;
            w = h * aspect;
        }
        let short_min = self.map_short_min as f64 * scale;
        let short = w.min(h);
        if short < short_min {
            let grow = short_min / short;
            w *= grow;
            h *= grow;
        }
        (even(w), even(h))
    }

    fn header_height(&self, rows: u8, scale: f64, font: f64) -> u32 {
        let header = &self.header;
        let meta_rows = u32::from(rows.saturating_sub(1));
        px(header.pad_top, scale)
            + px(header.title_row_h, scale * font)
            + meta_rows * px(header.meta_row_h, scale * font)
            + px(header.pad_bottom, scale)
    }

    fn header_rows_for(&self, content_w: u32, scale: f64, font: f64) -> u8 {
        let needed = self.header.two_row_min_content_w as f64 * scale * font;
        if content_w as f64 >= needed { 2 } else { 3 }
    }

    /// The canvas for one map of a grid whose projected aspect (width over
    /// height) is `grid_aspect`.
    pub fn plan_map(&self, grid_aspect: f64, class: SizeClass, user_scale: f64) -> CanvasPlan {
        let row = self.size_class(class);
        let scale = row.scale * if user_scale.is_finite() && user_scale > 0.0 { user_scale } else { 1.0 };
        let font = row.font;
        let aspect = self.clamp_aspect(grid_aspect);
        let clamped = grid_aspect.is_finite() && grid_aspect > 0.0 && aspect != grid_aspect;
        let (map_w, map_h) = self.map_size(aspect, self.map_area_px * scale * scale, scale);
        let margin = px(self.margin, scale);
        let bar_side = if aspect >= self.bar.bottom_min_aspect {
            BarSide::Bottom
        } else {
            BarSide::Right
        };
        let right = &self.bar.right;
        let bottom = &self.bar.bottom;
        let right_gutter = px(right.gap, scale)
            + px(right.thickness, scale)
            + px(right.tick_gap, scale)
            + px(right.label_w, scale * font);
        let bottom_footer = px(bottom.gap, scale)
            + px(bottom.thickness, scale)
            + px(bottom.tick_gap, scale)
            + px(bottom.label_h, scale * font);
        let group_w = match bar_side {
            BarSide::Right => map_w + right_gutter,
            BarSide::Bottom => map_w,
        };
        let canvas_w = (2 * margin + group_w).max(px(self.canvas_min_w, scale));
        let group_x = (canvas_w - group_w) / 2;
        let content_w = canvas_w - 2 * margin;
        let rows = self.header_rows_for(content_w, scale, font);
        let header_h = self.header_height(rows, scale, font);
        let map = PlanRect {
            x: group_x,
            y: header_h,
            w: map_w,
            h: map_h,
        };
        let (bar, canvas_h, bar_label_span, bar_tick_gap) = match bar_side {
            BarSide::Right => (
                PlanRect {
                    x: map.right() + px(right.gap, scale),
                    y: map.y,
                    w: px(right.thickness, scale),
                    h: map_h,
                },
                header_h + map_h + margin,
                px(right.label_w, scale * font),
                px(right.tick_gap, scale),
            ),
            BarSide::Bottom => (
                PlanRect {
                    x: map.x,
                    y: map.bottom() + px(bottom.gap, scale),
                    w: map_w,
                    h: px(bottom.thickness, scale),
                },
                header_h + map_h + bottom_footer + margin,
                px(bottom.label_h, scale * font),
                px(bottom.tick_gap, scale),
            ),
        };
        let tick_spacing = match bar_side {
            BarSide::Right => self.bar.tick_spacing_vertical_px,
            BarSide::Bottom => self.bar.tick_spacing_horizontal_px,
        };
        let text = (scale * font) as f32;
        CanvasPlan {
            canvas_w,
            canvas_h,
            map,
            header: PlanRect {
                x: margin,
                y: 0,
                w: content_w,
                h: header_h,
            },
            header_rows: rows,
            pad_top: px(self.header.pad_top, scale),
            title_row_h: px(self.header.title_row_h, scale * font),
            meta_row_h: px(self.header.meta_row_h, scale * font),
            bar: Some(bar),
            bar_side,
            bar_label_span,
            bar_tick_gap,
            title_px: self.header.title_px * text,
            title_min_px: self.header.title_min_px * text,
            meta_px: self.header.meta_px * text,
            tick_px: self.bar.tick_px * text,
            tick_spacing_px: px(tick_spacing, scale),
            grid_aspect,
            map_aspect: map_w as f64 / map_h as f64,
            aspect_clamped: clamped,
            scale,
            barb_spacing_px: self.barb_spacing_px * scale,
            outline_px: self.map_frame.outline_px * scale as f32,
            sharp_cell_px: self.sharp_cell_px,
        }
    }

    /// A sheet of `n` maps of one grid.  Rows and columns are chosen by
    /// score: how far the sheet's shape is from the table's target aspect,
    /// plus a weight per empty cell.
    pub fn plan_sheet(&self, n: usize, grid_aspect: f64, class: SizeClass, user_scale: f64) -> SheetPlan {
        let n = n.max(1) as u32;
        let row = self.size_class(class);
        let scale = row.scale * if user_scale.is_finite() && user_scale > 0.0 { user_scale } else { 1.0 };
        let font = row.font;
        let aspect = self.clamp_aspect(grid_aspect);
        let sheet = &self.sheet;
        let margin = px(self.margin, scale);
        let gutter = px(sheet.gutter, scale);
        let label_h = px(sheet.label_h, scale * font);
        let width_max = px(sheet.width_max, scale);
        let mut best: Option<(f64, SheetPlan)> = None;
        for columns in 1..=n {
            let rows = n.div_ceil(columns);
            let empty = rows * columns - n;
            let plan = self.sheet_with_grid(
                n, rows, columns, aspect, grid_aspect, scale, font, margin, gutter, label_h, width_max,
            );
            let shape = plan.canvas_w as f64 / plan.canvas_h as f64;
            let score = (shape / sheet.target_aspect).ln().abs() + sheet.empty_cell_weight * empty as f64;
            if best.as_ref().is_none_or(|(best_score, _)| score < *best_score - 1e-9) {
                best = Some((score, plan));
            }
        }
        best.expect("at least one sheet shape is scored").1
    }

    #[allow(clippy::too_many_arguments)]
    fn sheet_with_grid(
        &self,
        n: u32,
        rows: u32,
        columns: u32,
        aspect: f64,
        grid_aspect: f64,
        scale: f64,
        font: f64,
        margin: u32,
        gutter: u32,
        label_h: u32,
        width_max: u32,
    ) -> SheetPlan {
        let sheet = &self.sheet;
        let per_map = sheet.map_area_px * scale * scale / n as f64;
        let mut cell_w = (per_map * aspect).sqrt();
        let mut cell_h = cell_w / aspect;
        // The bar goes under a sheet that is wide overall, beside one that
        // is tall, by the same rule a single map uses.
        let grid_shape = (columns as f64 * cell_w) / (rows as f64 * (cell_h + label_h as f64));
        let bar_side = if grid_shape >= self.bar.bottom_min_aspect {
            BarSide::Bottom
        } else {
            BarSide::Right
        };
        let right = &self.bar.right;
        let bottom = &self.bar.bottom;
        let right_gutter = px(right.gap, scale)
            + px(right.thickness, scale)
            + px(right.tick_gap, scale)
            + px(right.label_w, scale * font);
        let side = match bar_side {
            BarSide::Right => right_gutter,
            BarSide::Bottom => 0,
        };
        let room = width_max.saturating_sub(2 * margin + side + gutter * (columns - 1)) as f64;
        if columns as f64 * cell_w > room {
            cell_w = room / columns as f64;
            cell_h = cell_w / aspect;
        }
        // Down to even pixels, so a capped sheet stays inside its cap.
        let (cell_w, cell_h) = (
            ((cell_w / 2.0).floor() * 2.0).max(2.0) as u32,
            ((cell_h / 2.0).floor() * 2.0).max(2.0) as u32,
        );
        let grid_w = columns * cell_w + gutter * (columns - 1);
        let canvas_w = (2 * margin + grid_w + side).max(px(self.canvas_min_w, scale));
        let content_w = canvas_w - 2 * margin;
        let header_rows = self.header_rows_for(content_w, scale, font);
        let header_h = self.header_height(header_rows, scale, font);
        let grid_x = (canvas_w - grid_w - side) / 2;
        let row_h = label_h + cell_h;
        let grid_h = rows * row_h + gutter * (rows - 1);
        let mut cells = Vec::with_capacity(n as usize);
        let mut labels = Vec::with_capacity(n as usize);
        for index in 0..n {
            let (r, c) = (index / columns, index % columns);
            let x = grid_x + c * (cell_w + gutter);
            let y = header_h + r * (row_h + gutter);
            labels.push(PlanRect { x, y, w: cell_w, h: label_h });
            cells.push(PlanRect { x, y: y + label_h, w: cell_w, h: cell_h });
        }
        let (bar, canvas_h) = match bar_side {
            BarSide::Right => (
                PlanRect {
                    x: grid_x + grid_w + px(right.gap, scale),
                    y: header_h + label_h,
                    w: px(right.thickness, scale),
                    h: grid_h - label_h,
                },
                header_h + grid_h + margin,
            ),
            BarSide::Bottom => (
                PlanRect {
                    x: grid_x,
                    y: header_h + grid_h + px(bottom.gap, scale),
                    w: grid_w,
                    h: px(bottom.thickness, scale),
                },
                header_h
                    + grid_h
                    + px(bottom.gap, scale)
                    + px(bottom.thickness, scale)
                    + px(bottom.tick_gap, scale)
                    + px(bottom.label_h, scale * font)
                    + margin,
            ),
        };
        let text = (scale * font) as f32;
        let member = CanvasPlan {
            canvas_w: cell_w,
            canvas_h: cell_h,
            map: PlanRect { x: 0, y: 0, w: cell_w, h: cell_h },
            header: PlanRect::default(),
            header_rows: 0,
            pad_top: 0,
            title_row_h: 0,
            meta_row_h: 0,
            bar: None,
            bar_side,
            bar_label_span: 0,
            bar_tick_gap: 0,
            title_px: self.header.title_px * text,
            title_min_px: self.header.title_min_px * text,
            meta_px: self.header.meta_px * text,
            tick_px: self.bar.tick_px * text,
            tick_spacing_px: 0,
            grid_aspect,
            map_aspect: cell_w as f64 / cell_h as f64,
            aspect_clamped: aspect != grid_aspect,
            scale,
            barb_spacing_px: self.barb_spacing_px * scale,
            outline_px: self.map_frame.outline_px * scale as f32,
            sharp_cell_px: self.sharp_cell_px,
        };
        let (bar_label_span, bar_tick_gap) = match bar_side {
            BarSide::Right => (px(right.label_w, scale * font), px(right.tick_gap, scale)),
            BarSide::Bottom => (px(bottom.label_h, scale * font), px(bottom.tick_gap, scale)),
        };
        let tick_spacing = match bar_side {
            BarSide::Right => self.bar.tick_spacing_vertical_px,
            BarSide::Bottom => self.bar.tick_spacing_horizontal_px,
        };
        let frame = CanvasPlan {
            canvas_w,
            canvas_h,
            map: PlanRect { x: grid_x, y: header_h, w: grid_w, h: grid_h },
            header: PlanRect { x: margin, y: 0, w: content_w, h: header_h },
            header_rows,
            pad_top: px(self.header.pad_top, scale),
            title_row_h: px(self.header.title_row_h, scale * font),
            meta_row_h: px(self.header.meta_row_h, scale * font),
            bar: Some(bar),
            bar_side,
            bar_label_span,
            bar_tick_gap,
            tick_spacing_px: px(tick_spacing, scale),
            ..member
        };
        SheetPlan {
            canvas_w,
            canvas_h,
            rows,
            columns,
            header: frame.header,
            header_rows,
            cells,
            labels,
            label_px: sheet.label_px * text,
            bar: Some(bar),
            bar_side,
            member,
            frame,
        }
    }

    /// The map header's geometry across a canvas of any size: what a
    /// section or any other non-map frame draws its header with, so every
    /// product family carries one header.  The map rectangle is the rest
    /// of the canvas; there is no bar.
    pub fn plan_header(&self, canvas_w: u32, canvas_h: u32, class: SizeClass, user_scale: f64) -> CanvasPlan {
        let mut plan = self.plan_map(1.0, class, user_scale);
        let margin = px(self.margin, plan.scale);
        let content_w = canvas_w.saturating_sub(2 * margin);
        let font = self.size_class(class).font;
        let rows = self.header_rows_for(content_w, plan.scale, font);
        let header_h = self.header_height(rows, plan.scale, font);
        plan.canvas_w = canvas_w;
        plan.canvas_h = canvas_h;
        plan.header = PlanRect { x: margin, y: 0, w: content_w, h: header_h };
        plan.header_rows = rows;
        plan.map = PlanRect { x: margin, y: header_h, w: content_w, h: canvas_h.saturating_sub(header_h) };
        plan.bar = None;
        plan
    }

    /// A vertical section's canvas: the plot area at the table's fixed
    /// aspect and budget, the axis gutters, the map header above and a
    /// bottom bar below.
    pub fn plan_section(&self, class: SizeClass, user_scale: f64) -> SectionPlan {
        let row = self.size_class(class);
        let scale = row.scale * if user_scale.is_finite() && user_scale > 0.0 { user_scale } else { 1.0 };
        let font = row.font;
        let section = &self.section;
        let plot_w = even((section.plot_area_px * scale * scale * section.plot_aspect).sqrt());
        let plot_h = even(plot_w as f64 / section.plot_aspect);
        let margin = px(self.margin, scale);
        let axis_left = px(section.axis_left, scale * font);
        let axis_bottom = px(section.axis_bottom, scale * font);
        let canvas_w = 2 * margin + axis_left + plot_w;
        let header_h = self.header_height(2, scale, font);
        let bottom = &self.bar.bottom;
        let footer = px(bottom.gap, scale)
            + px(bottom.thickness, scale)
            + px(bottom.tick_gap, scale)
            + px(bottom.label_h, scale * font);
        SectionPlan {
            canvas_w,
            canvas_h: header_h + plot_h + axis_bottom + footer + margin,
            plot: PlanRect {
                x: margin + axis_left,
                y: header_h,
                w: plot_w,
                h: plot_h,
            },
        }
    }
}

/// How the canvas is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum LayoutMode {
    /// The caller's pixel size, the map fitted inside it: today's look,
    /// kept for callers that tile at fixed pixels.
    Fixed,
    /// The canvas sized from the grid's shape through the layout table.
    Auto { class: SizeClass, scale: f64 },
}

impl Default for LayoutMode {
    fn default() -> Self {
        Self::Fixed
    }
}

static LAYOUT_MODE: RwLock<LayoutMode> = RwLock::new(LayoutMode::Fixed);
static PLANS: RwLock<Vec<CanvasPlan>> = RwLock::new(Vec::new());

/// Set the process-wide layout mode.  One invocation renders one run, so
/// this is set once from the command line, the same way the theme is.
pub fn set_layout_mode(mode: LayoutMode) {
    if let Ok(mut slot) = LAYOUT_MODE.write() {
        *slot = mode;
    }
}

pub fn layout_mode() -> LayoutMode {
    LAYOUT_MODE.read().map(|slot| *slot).unwrap_or_default()
}

pub fn auto_layout_active() -> bool {
    matches!(layout_mode(), LayoutMode::Auto { .. })
}

/// Make `plan` the layout for every render at its canvas size.  The
/// renderer looks a plan up by the canvas it is asked to draw, so every
/// lane that already carries a width and a height (the direct, derived,
/// windowed and generic lanes, and each member of a sheet) draws the
/// planned geometry without a new argument through each of them.
pub fn register_canvas_plan(plan: CanvasPlan) {
    if let Ok(mut plans) = PLANS.write() {
        plans.retain(|existing| {
            (existing.canvas_w, existing.canvas_h) != (plan.canvas_w, plan.canvas_h)
        });
        plans.push(plan);
    }
}

/// The plan registered for a canvas of exactly `width x height`, or for
/// an integer fraction of it (the supersample pass draws at a multiple).
pub fn canvas_plan_for(width: u32, height: u32) -> Option<CanvasPlan> {
    let plans = PLANS.read().ok()?;
    if plans.is_empty() {
        return None;
    }
    for plan in plans.iter() {
        if plan.canvas_w == width && plan.canvas_h == height {
            return Some(*plan);
        }
    }
    for plan in plans.iter() {
        if plan.canvas_w > 0
            && width % plan.canvas_w == 0
            && plan.canvas_h > 0
            && height % plan.canvas_h == 0
        {
            let factor = width / plan.canvas_w;
            if factor > 1 && height / plan.canvas_h == factor {
                return Some(plan.scaled(factor));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The suite's shapes (grid aspect from the survey domains).
    const SUITE: [(&str, f64); 7] = [
        ("square", 1.0),
        ("metro-d01", 1.06),
        ("nest-d02", 1.16),
        ("tiny", 1.23),
        ("wide-conus", 1.75),
        ("wide", 2.91),
        ("tall", 0.399),
    ];

    fn table() -> &'static LayoutTable {
        LayoutTable::builtin()
    }

    #[test]
    fn layers_follow_the_frame_size_not_its_location() {
        let layers = &table().layers;
        assert!(layers.counties_visible(300.0));
        assert!(layers.counties_visible(900.0));
        assert!(!layers.counties_visible(3000.0), "no county mesh at continental scale");
        assert_eq!(layers.place_tier(80.0), "dense");
        assert_eq!(layers.place_tier(340.0), "major_and_aux");
        assert_eq!(layers.place_tier(3000.0), "major");
    }

    #[test]
    fn the_compiled_in_table_parses() {
        let table = table();
        assert!(table.map_area_px > 0.0);
        assert!(table.aspect_min < 1.0 && table.aspect_max > 1.0);
    }

    #[test]
    fn a_square_grid_gets_a_square_map_and_a_bar_at_map_height() {
        let plan = table().plan_map(1.0, SizeClass::Standard, 1.0);
        assert_eq!((plan.map.w, plan.map.h), (774, 774));
        assert_eq!(plan.bar_side, BarSide::Right);
        let bar = plan.bar.unwrap();
        assert_eq!((bar.y, bar.h), (plan.map.y, plan.map.h), "bar is exactly the map's height");
        assert_eq!(plan.header_rows, 2);
        assert_eq!((plan.canvas_w, plan.canvas_h), (888, 866));
    }

    #[test]
    fn every_suite_shape_draws_its_map_at_the_grid_aspect_inside_the_band() {
        for (name, aspect) in SUITE {
            let plan = table().plan_map(aspect, SizeClass::Standard, 1.0);
            let clamped = table().clamp_aspect(aspect);
            let drawn = plan.map.w as f64 / plan.map.h as f64;
            assert!(
                (drawn - clamped).abs() / clamped < 0.01,
                "{name}: map {}x{} is {drawn:.3}, grid {aspect}",
                plan.map.w,
                plan.map.h
            );
            let area = (plan.map.w * plan.map.h) as f64;
            assert!(area <= table().map_area_px * 1.02, "{name}: {area} px over the budget");
        }
    }

    /// The layout law this module exists for: no image edge is farther
    /// from the map than the chrome margin plus what the chrome itself
    /// needs, on every suite shape.
    #[test]
    fn no_canvas_edge_is_more_than_the_chrome_away_from_the_map() {
        let t = table();
        let right_gutter =
            t.bar.right.gap + t.bar.right.thickness + t.bar.right.tick_gap + t.bar.right.label_w;
        let bottom_footer =
            t.bar.bottom.gap + t.bar.bottom.thickness + t.bar.bottom.tick_gap + t.bar.bottom.label_h;
        for (name, aspect) in SUITE {
            let plan = t.plan_map(aspect, SizeClass::Standard, 1.0);
            let left = plan.map.x;
            let right = plan.canvas_w - plan.map.right();
            let bottom = plan.canvas_h - plan.map.bottom();
            let top = plan.map.y;
            let (right_allow, bottom_allow) = match plan.bar_side {
                BarSide::Right => (t.margin + right_gutter, t.margin),
                BarSide::Bottom => (t.margin, t.margin + bottom_footer),
            };
            // The canvas floor may widen a very tall frame; the map group
            // is then centred, never pushed to one side.
            let floor_pad = (t.canvas_min_w as i64
                - (2 * t.margin + plan.map.w + if plan.bar_side == BarSide::Right { right_gutter } else { 0 }) as i64)
                .max(0) as u32;
            assert!(left <= t.margin + floor_pad / 2 + 1, "{name}: {left} px left of the map");
            assert!(right <= right_allow + floor_pad / 2 + 1, "{name}: {right} px right of the map");
            assert!(bottom <= bottom_allow, "{name}: {bottom} px under the map");
            assert!(top == plan.header.h, "{name}: header {} but map at {top}", plan.header.h);
        }
    }

    #[test]
    fn the_colour_bar_is_inside_the_canvas_and_matches_the_map_side() {
        for (name, aspect) in SUITE {
            let plan = table().plan_map(aspect, SizeClass::Standard, 1.0);
            let bar = plan.bar.unwrap();
            match plan.bar_side {
                BarSide::Right => {
                    assert_eq!((bar.y, bar.h), (plan.map.y, plan.map.h), "{name}");
                    assert!(bar.right() + plan.bar_tick_gap + plan.bar_label_span <= plan.canvas_w, "{name}");
                }
                BarSide::Bottom => {
                    assert_eq!((bar.x, bar.w), (plan.map.x, plan.map.w), "{name}");
                    assert!(bar.bottom() + plan.bar_tick_gap + plan.bar_label_span <= plan.canvas_h, "{name}");
                }
            }
        }
    }

    #[test]
    fn wide_frames_take_a_bottom_bar_and_tall_frames_three_header_rows() {
        let wide = table().plan_map(2.91, SizeClass::Standard, 1.0);
        assert_eq!(wide.bar_side, BarSide::Bottom);
        assert_eq!(wide.header_rows, 2);
        let tall = table().plan_map(0.399, SizeClass::Standard, 1.0);
        assert_eq!(tall.bar_side, BarSide::Right);
        assert_eq!(tall.header_rows, 3);
        assert!(tall.aspect_clamped);
        assert_eq!((tall.map.w, tall.map.h), (400, 1000));
    }

    #[test]
    fn the_canvas_depends_only_on_the_domain_so_loops_do_not_jitter() {
        let a = table().plan_map(1.23, SizeClass::Standard, 1.0);
        let b = table().plan_map(1.23, SizeClass::Standard, 1.0);
        assert_eq!(a, b);
        assert_eq!(a.without_bar().canvas_w, a.canvas_w);
        assert_eq!(a.without_bar().map, a.map);
    }

    #[test]
    fn size_classes_scale_pixels_and_areas() {
        let standard = table().plan_map(1.0, SizeClass::Standard, 1.0);
        let large = table().plan_map(1.0, SizeClass::Large, 1.0);
        assert!((large.map.w as f64 / standard.map.w as f64 - 2.0).abs() < 0.01);
        let phone = table().plan_map(1.0, SizeClass::Phone, 1.0);
        assert!(phone.map.w < standard.map.w);
        assert!(phone.meta_px > standard.meta_px * 0.8, "phone text is enlarged, not only shrunk");
    }

    #[test]
    fn a_degenerate_aspect_plans_as_a_square() {
        let plan = table().plan_map(f64::NAN, SizeClass::Standard, 1.0);
        assert_eq!(plan.map.w, plan.map.h);
    }

    #[test]
    fn a_three_panel_sheet_of_a_square_grid_lays_out_in_one_row() {
        let sheet = table().plan_sheet(3, 1.0, SizeClass::Standard, 1.0);
        assert_eq!((sheet.rows, sheet.columns), (1, 3));
        assert_eq!(sheet.cells.len(), 3);
        assert_eq!(sheet.bar_side, BarSide::Bottom);
        assert!(sheet.canvas_w <= table().sheet.width_max);
        for (cell, label) in sheet.cells.iter().zip(&sheet.labels) {
            assert_eq!(label.bottom(), cell.y, "each label strip sits directly on its map");
            assert!(cell.right() <= sheet.canvas_w && cell.bottom() <= sheet.canvas_h);
            assert_eq!((cell.w, cell.h), (sheet.member.canvas_w, sheet.member.canvas_h));
        }
        let bar = sheet.bar.unwrap();
        assert_eq!(bar.x, sheet.cells[0].x);
        assert_eq!(bar.right(), sheet.cells[2].right());
    }

    #[test]
    fn a_four_panel_sheet_prefers_a_square_grid_to_one_long_row() {
        let sheet = table().plan_sheet(4, 1.0, SizeClass::Standard, 1.0);
        assert_eq!((sheet.rows, sheet.columns), (2, 2));
    }

    #[test]
    fn a_section_is_a_two_to_one_plot_under_the_map_header() {
        let plan = table().plan_section(SizeClass::Standard, 1.0);
        assert_eq!(plan.plot.w, 2 * plan.plot.h);
        assert!(plan.plot.right() < plan.canvas_w);
        assert!(plan.plot.bottom() < plan.canvas_h);
    }

    #[test]
    fn a_registered_plan_is_found_at_its_canvas_and_at_a_supersample_multiple() {
        let mut plan = table().plan_map(1.37, SizeClass::Standard, 1.0);
        // A size no other test renders at.
        plan.canvas_w = 1001;
        plan.canvas_h = 667;
        register_canvas_plan(plan);
        assert_eq!(canvas_plan_for(1001, 667), Some(plan));
        let doubled = canvas_plan_for(2002, 1334).expect("supersampled plan");
        assert_eq!(doubled.map.w, plan.map.w * 2);
        assert_eq!(canvas_plan_for(1001, 668), None);
    }
}
