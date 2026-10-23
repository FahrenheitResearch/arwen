//! The full SPC sounding window: skew-T + speed/advection strips, hodograph
//! with locator inset, storm slinky / theta-e / SR-winds / hazard row, and
//! the bottom index-board band — laid out like the vendored `SPCWidget` grid
//! with the SHARPpy-Reimagined modifications (see PORTING.md).

use egui::{Align2, Rect, Response, Sense, Ui, Vec2, Widget};

use crate::derived::DerivedParams;
use crate::diagnostic_table::{
    DiagnosticTableBoard, DiagnosticTablePanelKind, NativeDiagnosticPatchBoard,
};
use crate::panels;

/// What to draw in the fourth inset cell (upper-right row).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CornerPanel {
    /// Location map with the sounding point (default).
    #[default]
    LocationMap,
    /// The original SHARPpy "Psbl Haz. Type" watch box.
    HazardType,
}

/// Every swappable panel of the window. Any cell (except the skew-T) can
/// hold any of these, or be hidden.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelKind {
    Speed,
    Advection,
    Hodograph,
    Slinky,
    ThetaE,
    SrWinds,
    LocationMap,
    HazardType,
    /// Parcel/thermodynamic/lapse-rate portion of the legacy index board.
    ConvectiveIndices,
    /// Layer kinematics and storm-motion portion of the legacy index board.
    Kinematics,
    /// Environmental and severe-weather scalar portion of the legacy index
    /// board (the SHIP distribution is its own panel).
    SevereIndices,
    /// Historical combined three-column index board, retained for restored
    /// custom layouts.
    IndexBoard,
    Ship,
    Streamwiseness,
    Stp,
    Hidden,
}

impl PanelKind {
    pub const ALL: [PanelKind; 16] = [
        PanelKind::Speed,
        PanelKind::Advection,
        PanelKind::Hodograph,
        PanelKind::Slinky,
        PanelKind::ThetaE,
        PanelKind::SrWinds,
        PanelKind::LocationMap,
        PanelKind::HazardType,
        PanelKind::ConvectiveIndices,
        PanelKind::Kinematics,
        PanelKind::SevereIndices,
        PanelKind::IndexBoard,
        PanelKind::Ship,
        PanelKind::Streamwiseness,
        PanelKind::Stp,
        PanelKind::Hidden,
    ];

    pub fn label(self) -> &'static str {
        match self {
            PanelKind::Speed => "Wind speed",
            PanelKind::Advection => "Temp advection",
            PanelKind::Hodograph => "Hodograph",
            PanelKind::Slinky => "Storm slinky",
            PanelKind::ThetaE => "Theta-E v. pres",
            PanelKind::SrWinds => "SR wind v. height",
            PanelKind::LocationMap => "Location map",
            PanelKind::HazardType => "Psbl haz. type",
            PanelKind::ConvectiveIndices => "Parcels & thermo",
            PanelKind::Kinematics => "Kinematics",
            PanelKind::SevereIndices => "Severe indices",
            PanelKind::IndexBoard => "Combined index board",
            PanelKind::Ship => "SHIP box",
            PanelKind::Streamwiseness => "Streamwiseness",
            PanelKind::Stp => "Effective STP",
            PanelKind::Hidden => "(hidden)",
        }
    }

    /// Stable serialization token for this panel (see
    /// [`SoundingLayout::to_tokens`]). Lowercase, no separators; new
    /// variants get new tokens and existing tokens never change.
    pub fn token(self) -> &'static str {
        match self {
            PanelKind::Speed => "speed",
            PanelKind::Advection => "advection",
            PanelKind::Hodograph => "hodograph",
            PanelKind::Slinky => "slinky",
            PanelKind::ThetaE => "thetae",
            PanelKind::SrWinds => "srwinds",
            PanelKind::LocationMap => "locationmap",
            PanelKind::HazardType => "hazardtype",
            PanelKind::ConvectiveIndices => "convectiveindices",
            PanelKind::Kinematics => "kinematics",
            PanelKind::SevereIndices => "severeindices",
            PanelKind::IndexBoard => "indexboard",
            PanelKind::Ship => "ship",
            PanelKind::Streamwiseness => "streamwiseness",
            PanelKind::Stp => "stp",
            PanelKind::Hidden => "hidden",
        }
    }

    /// Inverse of [`PanelKind::token`]; `None` for unknown tokens.
    pub fn from_token(token: &str) -> Option<PanelKind> {
        PanelKind::ALL.into_iter().find(|k| k.token() == token)
    }

    fn draw(
        self,
        painter: &egui::Painter,
        rect: Rect,
        prof: &Profile,
        dv: &DerivedParams,
        st: &SkewTStyle,
        hodo_zoom: f64,
    ) {
        match self {
            PanelKind::Speed => panels::speed::draw(painter, rect, prof, dv, st),
            PanelKind::Advection => panels::advection::draw(painter, rect, prof, dv, st),
            PanelKind::Hodograph => {
                panels::hodo::draw_zoomed(painter, rect, prof, dv, st, hodo_zoom)
            }
            PanelKind::Slinky => panels::slinky::draw(painter, rect, prof, dv, st),
            PanelKind::ThetaE => panels::thetae::draw(painter, rect, prof, dv, st),
            PanelKind::SrWinds => panels::srwinds::draw(painter, rect, prof, dv, st),
            PanelKind::LocationMap => panels::locator::draw(painter, rect, prof, dv, st),
            PanelKind::HazardType => panels::hazard::draw(painter, rect, prof, dv, st),
            PanelKind::ConvectiveIndices => {
                panels::index_board::draw_convective(painter, rect, prof, dv, st)
            }
            PanelKind::Kinematics => {
                panels::index_board::draw_kinematics(painter, rect, prof, dv, st)
            }
            PanelKind::SevereIndices => {
                panels::index_board::draw_indices(painter, rect, prof, dv, st)
            }
            PanelKind::IndexBoard => panels::index_board::draw(painter, rect, prof, dv, st),
            PanelKind::Ship => panels::ship_inset::draw(painter, rect, prof, dv, st),
            PanelKind::Streamwiseness => panels::streamwiseness::draw(painter, rect, prof, dv, st),
            PanelKind::Stp => panels::stp::draw(painter, rect, prof, dv, st),
            PanelKind::Hidden => {}
        }
    }
}

/// User-adjustable window layout: which panel lives in each cell of the SPC
/// grid (the skew-T cell is fixed), plus the hodograph zoom. Kept in egui
/// memory per widget id; edited in-app via the gear button.
#[derive(Clone, Debug, PartialEq)]
pub struct SoundingLayout {
    /// The two narrow strips right of the skew-T.
    pub strips: [PanelKind; 2],
    /// The large upper-right cell.
    pub main: PanelKind,
    /// The four inset cells under it.
    pub insets: [PanelKind; 4],
    /// The six bottom-band cells. Slots 0 and 1 occupy full-height columns;
    /// slots 2 and 3 share the third column vertically; slots 4 and 5 occupy
    /// full-height columns. This keeps SHIP and the scalar severe indices
    /// independently movable without changing the familiar board silhouette.
    pub bottom: [PanelKind; 6],
    /// Hodograph window width (kts across).
    pub hodo_zoom_kts: f64,
    /// Height occupied by the upper (skew-T + right diagnostic grid) row,
    /// expressed as a fraction of the complete sounding board.
    pub top_height_fraction: f32,
    /// Width occupied by the skew-T, expressed as a fraction of the complete
    /// sounding board. The right diagnostic grid receives the remainder.
    pub skew_width_fraction: f32,
    /// Height occupied by the large panel in the right diagnostic grid,
    /// expressed as a fraction of that grid below the brand strip. The inset
    /// row receives the remainder.
    pub right_main_height_fraction: f32,
    /// Width fractions for the two narrow strips and the remaining large
    /// right-grid area, respectively.
    pub right_column_fractions: [f32; 3],
    /// Width fractions for the four inset cells under the large right panel.
    pub inset_column_fractions: [f32; 4],
    /// Width fractions for the five bottom-band columns. A column whose slots
    /// are all hidden retains its saved fraction but surrenders it to visible
    /// columns while hidden.
    pub bottom_column_fractions: [f32; 5],
    /// Height fraction of the upper cell in the split third bottom column.
    pub bottom_split_fraction: f32,
}

const DEFAULT_TOP_HEIGHT_FRACTION: f32 = 0.67;
const DEFAULT_SKEW_WIDTH_FRACTION: f32 = 0.46;
const DEFAULT_RIGHT_MAIN_HEIGHT_FRACTION: f32 = 8.0 / 11.0;
const DEFAULT_RIGHT_COLUMN_FRACTIONS: [f32; 3] = [3.0 / 29.0, 2.0 / 29.0, 24.0 / 29.0];
const DEFAULT_INSET_COLUMN_FRACTIONS: [f32; 4] = [0.25; 4];
// The first three columns exactly partition the historical IndexBoard share
// (0.61) at its old 38% / 33.8% / 28.2% internal boundaries. Streamwiseness
// and the optional final cell keep their previous 0.14 / 0.25 shares.
const DEFAULT_BOTTOM_COLUMN_FRACTIONS: [f32; 5] = [0.2318, 0.20618, 0.17202, 0.14, 0.25];
const DEFAULT_BOTTOM_SPLIT_FRACTION: f32 = 0.51;
const LEGACY_DEFAULT_BOTTOM_COLUMN_FRACTIONS: [f32; 3] = [0.61, 0.14, 0.25];

const MIN_TOP_HEIGHT_FRACTION: f32 = 0.40;
const MAX_TOP_HEIGHT_FRACTION: f32 = 0.85;
const MIN_SKEW_WIDTH_FRACTION: f32 = 0.30;
const MAX_SKEW_WIDTH_FRACTION: f32 = 0.70;
const MIN_RIGHT_MAIN_HEIGHT_FRACTION: f32 = 0.35;
const MAX_RIGHT_MAIN_HEIGHT_FRACTION: f32 = 0.85;
const MIN_BOTTOM_SPLIT_FRACTION: f32 = 0.20;
const MAX_BOTTOM_SPLIT_FRACTION: f32 = 0.80;
const MIN_TRACK_FRACTION: f32 = 0.05;
const GEOMETRY_TOKEN_PREFIX: &str = "g1:";
const SPLIT_BOARD_GEOMETRY_TOKEN_PREFIX: &str = "g2:";

fn migrate_legacy_bottom(
    panels: [PanelKind; 3],
    fractions: [f32; 3],
) -> ([PanelKind; 6], [f32; 5]) {
    if panels[0] == PanelKind::IndexBoard {
        (
            [
                PanelKind::ConvectiveIndices,
                PanelKind::Kinematics,
                PanelKind::Ship,
                PanelKind::SevereIndices,
                panels[1],
                panels[2],
            ],
            [
                fractions[0] * 0.38,
                fractions[0] * 0.338,
                fractions[0] * 0.282,
                fractions[1],
                fractions[2],
            ],
        )
    } else {
        // Preserve arbitrary old three-cell layouts as three full-height
        // columns. The two hidden spacer columns surrender their allocation,
        // so the rendered old-panel width ratios remain unchanged.
        (
            [
                panels[0],
                panels[1],
                PanelKind::Hidden,
                PanelKind::Hidden,
                panels[2],
                PanelKind::Hidden,
            ],
            [
                fractions[0],
                fractions[1],
                MIN_TRACK_FRACTION,
                fractions[2],
                MIN_TRACK_FRACTION,
            ],
        )
    }
}

impl Default for SoundingLayout {
    fn default() -> Self {
        SoundingLayout {
            strips: [PanelKind::Speed, PanelKind::Advection],
            main: PanelKind::Hodograph,
            insets: [
                PanelKind::Slinky,
                PanelKind::ThetaE,
                PanelKind::SrWinds,
                PanelKind::LocationMap,
            ],
            bottom: [
                PanelKind::ConvectiveIndices,
                PanelKind::Kinematics,
                PanelKind::Ship,
                PanelKind::SevereIndices,
                PanelKind::Streamwiseness,
                PanelKind::Hidden,
            ],
            hodo_zoom_kts: panels::hodo::DEFAULT_ZOOM_KTS,
            top_height_fraction: DEFAULT_TOP_HEIGHT_FRACTION,
            skew_width_fraction: DEFAULT_SKEW_WIDTH_FRACTION,
            right_main_height_fraction: DEFAULT_RIGHT_MAIN_HEIGHT_FRACTION,
            right_column_fractions: DEFAULT_RIGHT_COLUMN_FRACTIONS,
            inset_column_fractions: DEFAULT_INSET_COLUMN_FRACTIONS,
            bottom_column_fractions: DEFAULT_BOTTOM_COLUMN_FRACTIONS,
            bottom_split_fraction: DEFAULT_BOTTOM_SPLIT_FRACTION,
        }
    }
}

fn normalize_track_fractions<const N: usize>(fractions: &mut [f32; N]) {
    for value in fractions.iter_mut() {
        if !value.is_finite() || *value < 0.0 {
            *value = 0.0;
        }
    }
    let total: f32 = fractions.iter().sum();
    if !total.is_finite() || total <= f32::EPSILON {
        fractions.fill(1.0 / N as f32);
    } else if (total - 1.0).abs() > 1.0e-6 {
        for value in fractions.iter_mut() {
            *value /= total;
        }
    }

    // Give every track a durable grab target. Take a short track's deficit
    // only from the other tracks' space above the same floor, so already-valid
    // weights (including the pixel-matching defaults) remain unchanged.
    let deficit: f32 = fractions
        .iter()
        .map(|value| (MIN_TRACK_FRACTION - *value).max(0.0))
        .sum();
    if deficit > 0.0 {
        let available: f32 = fractions
            .iter()
            .map(|value| (*value - MIN_TRACK_FRACTION).max(0.0))
            .sum();
        if available <= deficit {
            fractions.fill(1.0 / N as f32);
        } else {
            for value in fractions.iter_mut() {
                if *value < MIN_TRACK_FRACTION {
                    *value = MIN_TRACK_FRACTION;
                } else {
                    *value -= deficit * (*value - MIN_TRACK_FRACTION) / available;
                }
            }
        }
    }
}

impl SoundingLayout {
    /// Serialize to a compact, dependency-free token string a host can stash
    /// in its own settings (JSON, ini, ...). Format — five `|`-separated
    /// sections, panel tokens comma-separated within a section:
    ///
    /// ```text
    /// strips(2) | main(1) | insets(4) | bottom(6) | hodo_zoom_kts
    /// ```
    ///
    /// e.g. the default layout is
    /// `"speed,advection|hodograph|slinky,thetae,srwinds,locationmap|convectiveindices,kinematics,ship,severeindices,streamwiseness,hidden|250"`.
    /// Panel tokens come from [`PanelKind::token`]; the zoom is a plain
    /// decimal in knots. A non-default geometry appends a sixth, versioned
    /// section: `g2:` followed by the three major split fractions, the three
    /// right-column fractions, four inset fractions, five bottom-band column
    /// fractions, and the split-column height fraction (groups are separated
    /// by `;`). [`SoundingLayout::from_tokens`] also migrates the historical
    /// three-cell bottom section and `g1:` geometry.
    pub fn to_tokens(&self) -> String {
        let csv = |kinds: &[PanelKind]| {
            kinds
                .iter()
                .map(|k| k.token())
                .collect::<Vec<_>>()
                .join(",")
        };
        let float_csv = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        let mut layout = self.clone();
        layout.normalize_geometry();
        let mut tokens = format!(
            "{}|{}|{}|{}|{}",
            csv(&layout.strips),
            layout.main.token(),
            csv(&layout.insets),
            csv(&layout.bottom),
            layout.hodo_zoom_kts,
        );
        if !layout.has_default_geometry() {
            tokens.push('|');
            tokens.push_str(SPLIT_BOARD_GEOMETRY_TOKEN_PREFIX);
            tokens.push_str(&float_csv(&[
                layout.top_height_fraction,
                layout.skew_width_fraction,
                layout.right_main_height_fraction,
            ]));
            tokens.push(';');
            tokens.push_str(&float_csv(&layout.right_column_fractions));
            tokens.push(';');
            tokens.push_str(&float_csv(&layout.inset_column_fractions));
            tokens.push(';');
            tokens.push_str(&float_csv(&layout.bottom_column_fractions));
            tokens.push(';');
            tokens.push_str(&layout.bottom_split_fraction.to_string());
        }
        tokens
    }

    /// Parse a [`SoundingLayout::to_tokens`] string. Whitespace around
    /// tokens is tolerated; the zoom is clamped to the interactive range
    /// (80–500 kts). Returns `None` for wrong section/panel counts,
    /// unknown panel tokens, malformed geometry, or non-finite numbers. Old
    /// three-cell bottom sections are migrated: a leading historical combined
    /// index board expands into the new convective, kinematic, SHIP, and
    /// severe-index panels while the other two old cells keep their roles.
    pub fn from_tokens(s: &str) -> Option<SoundingLayout> {
        fn cells<const N: usize>(section: &str) -> Option<[PanelKind; N]> {
            let mut out = [PanelKind::Hidden; N];
            let mut it = section.split(',');
            for slot in &mut out {
                *slot = PanelKind::from_token(it.next()?.trim())?;
            }
            it.next().is_none().then_some(out)
        }
        fn floats<const N: usize>(section: &str) -> Option<[f32; N]> {
            let mut out = [0.0; N];
            let mut it = section.split(',');
            for slot in &mut out {
                let value: f32 = it.next()?.trim().parse().ok()?;
                if !value.is_finite() {
                    return None;
                }
                *slot = value;
            }
            it.next().is_none().then_some(out)
        }
        let mut sections = s.split('|');
        let strips = cells::<2>(sections.next()?)?;
        let [main] = cells::<1>(sections.next()?)?;
        let insets = cells::<4>(sections.next()?)?;
        let bottom_section = sections.next()?;
        let (bottom, legacy_bottom) = if let Some(bottom) = cells::<6>(bottom_section) {
            (bottom, None)
        } else {
            let legacy = cells::<3>(bottom_section)?;
            let (bottom, _) = migrate_legacy_bottom(legacy, LEGACY_DEFAULT_BOTTOM_COLUMN_FRACTIONS);
            (bottom, Some(legacy))
        };
        let zoom: f64 = sections.next()?.trim().parse().ok()?;
        let geometry = sections.next();
        if sections.next().is_some() || !zoom.is_finite() {
            return None;
        }
        let mut layout = SoundingLayout {
            strips,
            main,
            insets,
            bottom,
            hodo_zoom_kts: zoom.clamp(80.0, 500.0),
            ..SoundingLayout::default()
        };
        if let Some(legacy) = legacy_bottom {
            let (_, fractions) =
                migrate_legacy_bottom(legacy, LEGACY_DEFAULT_BOTTOM_COLUMN_FRACTIONS);
            layout.bottom_column_fractions = fractions;
        }
        if let Some(geometry) = geometry {
            let geometry = geometry.trim();
            let (version, body) = if let Some(body) = geometry.strip_prefix(GEOMETRY_TOKEN_PREFIX) {
                (1, body)
            } else if let Some(body) = geometry.strip_prefix(SPLIT_BOARD_GEOMETRY_TOKEN_PREFIX) {
                (2, body)
            } else {
                return None;
            };
            let mut groups = body.split(';');
            let [top_height, skew_width, right_main_height] = floats::<3>(groups.next()?)?;
            layout.top_height_fraction = top_height;
            layout.skew_width_fraction = skew_width;
            layout.right_main_height_fraction = right_main_height;
            layout.right_column_fractions = floats::<3>(groups.next()?)?;
            layout.inset_column_fractions = floats::<4>(groups.next()?)?;
            if version == 1 {
                let legacy_fractions = floats::<3>(groups.next()?)?;
                layout.bottom_column_fractions = if let Some(legacy) = legacy_bottom {
                    migrate_legacy_bottom(legacy, legacy_fractions).1
                } else {
                    // Be liberal with a hand-authored six-cell layout carrying
                    // old geometry: interpret the former IndexBoard share as
                    // the first three new columns.
                    [
                        legacy_fractions[0] * 0.38,
                        legacy_fractions[0] * 0.338,
                        legacy_fractions[0] * 0.282,
                        legacy_fractions[1],
                        legacy_fractions[2],
                    ]
                };
            } else {
                layout.bottom_column_fractions = floats::<5>(groups.next()?)?;
                let [split] = floats::<1>(groups.next()?)?;
                layout.bottom_split_fraction = split;
            }
            if groups.next().is_some() {
                return None;
            }
        }
        layout.normalize_geometry();
        Some(layout)
    }

    fn normalize_geometry(&mut self) {
        self.top_height_fraction = if self.top_height_fraction.is_finite() {
            self.top_height_fraction
        } else {
            DEFAULT_TOP_HEIGHT_FRACTION
        }
        .clamp(MIN_TOP_HEIGHT_FRACTION, MAX_TOP_HEIGHT_FRACTION);
        self.skew_width_fraction = if self.skew_width_fraction.is_finite() {
            self.skew_width_fraction
        } else {
            DEFAULT_SKEW_WIDTH_FRACTION
        }
        .clamp(MIN_SKEW_WIDTH_FRACTION, MAX_SKEW_WIDTH_FRACTION);
        self.right_main_height_fraction = if self.right_main_height_fraction.is_finite() {
            self.right_main_height_fraction
        } else {
            DEFAULT_RIGHT_MAIN_HEIGHT_FRACTION
        }
        .clamp(
            MIN_RIGHT_MAIN_HEIGHT_FRACTION,
            MAX_RIGHT_MAIN_HEIGHT_FRACTION,
        );
        self.bottom_split_fraction = if self.bottom_split_fraction.is_finite() {
            self.bottom_split_fraction
        } else {
            DEFAULT_BOTTOM_SPLIT_FRACTION
        }
        .clamp(MIN_BOTTOM_SPLIT_FRACTION, MAX_BOTTOM_SPLIT_FRACTION);
        normalize_track_fractions(&mut self.right_column_fractions);
        normalize_track_fractions(&mut self.inset_column_fractions);
        normalize_track_fractions(&mut self.bottom_column_fractions);
    }

    fn has_default_geometry(&self) -> bool {
        self.top_height_fraction == DEFAULT_TOP_HEIGHT_FRACTION
            && self.skew_width_fraction == DEFAULT_SKEW_WIDTH_FRACTION
            && self.right_main_height_fraction == DEFAULT_RIGHT_MAIN_HEIGHT_FRACTION
            && self.right_column_fractions == DEFAULT_RIGHT_COLUMN_FRACTIONS
            && self.inset_column_fractions == DEFAULT_INSET_COLUMN_FRACTIONS
            && self.bottom_column_fractions == DEFAULT_BOTTOM_COLUMN_FRACTIONS
            && self.bottom_split_fraction == DEFAULT_BOTTOM_SPLIT_FRACTION
    }
}

/// Read the [`SoundingLayout`] stored in egui temp memory under `id` — the
/// key a [`SoundingView`] built with [`SoundingView::layout_memory_id`]
/// reads and writes. `None` until something stored one.
pub fn stored_layout(ctx: &egui::Context, id: egui::Id) -> Option<SoundingLayout> {
    ctx.data_mut(|d| d.get_temp(id))
}

/// Store a [`SoundingLayout`] in egui temp memory under `id`, where a
/// [`SoundingView`] built with [`SoundingView::layout_memory_id`] of the
/// same `id` picks it up on its next frame. Together with
/// [`SoundingLayout::to_tokens`] / [`from_tokens`](SoundingLayout::from_tokens)
/// this lets a host persist the layout across sessions.
pub fn store_layout(ctx: &egui::Context, id: egui::Id, layout: &SoundingLayout) {
    let mut layout = layout.clone();
    layout.normalize_geometry();
    ctx.data_mut(|d| d.insert_temp(id, layout));
}
use crate::profile::{ParcelType, Profile};
use crate::skewt::{SkewT, SkewTStyle};

/// The complete sounding window. Compute [`DerivedParams`] once (it is not
/// cheap) and keep it alongside the profile.
pub struct SoundingView<'a> {
    prof: &'a Profile,
    derived: &'a DerivedParams,
    title: String,
    brand: Option<String>,
    parcel: ParcelType,
    style: SkewTStyle,
    size: Option<Vec2>,
    corner: CornerPanel,
    interactive: bool,
    layout_id: Option<egui::Id>,
    diagnostic_tables: Option<&'a DiagnosticTableBoard>,
    native_diagnostic_patches: Option<&'a NativeDiagnosticPatchBoard>,
    // website renderer patch: pass-through of the SkewT omega / wind-barb switches.
    plot_omega: Option<bool>,
    interp_winds: bool,
}

impl<'a> SoundingView<'a> {
    pub fn new(prof: &'a Profile, derived: &'a DerivedParams) -> Self {
        SoundingView {
            prof,
            derived,
            title: String::new(),
            brand: None,
            parcel: ParcelType::MostUnstable,
            style: SkewTStyle::default(),
            size: None,
            corner: CornerPanel::default(),
            interactive: true,
            layout_id: None,
            diagnostic_tables: None,
            native_diagnostic_patches: None,
            plot_omega: None,
            interp_winds: true,
        }
    }

    /// website renderer patch: force the skew-T omega meter on/off (default: on
    /// when omega data exists), as [`SkewT::plot_omega`].
    pub fn plot_omega(mut self, on: bool) -> Self {
        self.plot_omega = Some(on);
        self
    }

    /// website renderer patch: barbs every 40 hPa (true, default) or at raw levels,
    /// as [`SkewT::interp_winds`].
    pub fn interp_winds(mut self, on: bool) -> Self {
        self.interp_winds = on;
        self
    }

    /// Pin the egui-memory key the panel layout is kept under (default: an
    /// id derived from the widget's `ui.id()`, which shifts with the
    /// surrounding layout). With a stable id the host can read/write the
    /// layout via [`stored_layout`] / [`store_layout`] — e.g. to persist it
    /// with [`SoundingLayout::to_tokens`].
    pub fn layout_memory_id(mut self, id: egui::Id) -> Self {
        self.layout_id = Some(id);
        self
    }

    /// Replace any supplied scalar-table panels with host-resolved rows.
    /// Omitted panels retain their exact native renderer; without this call
    /// the complete sounding window is unchanged.
    pub fn diagnostic_tables(mut self, tables: &'a DiagnosticTableBoard) -> Self {
        self.diagnostic_tables = Some(tables);
        self
    }

    /// Apply sparse, display-ready replacements to native diagnostic cells
    /// without replacing the surrounding native table geometry.
    pub fn native_diagnostic_patches(mut self, patches: &'a NativeDiagnosticPatchBoard) -> Self {
        self.native_diagnostic_patches = Some(patches);
        self
    }

    /// Enable/disable the hover readout cursor and the linked hodograph
    /// marker (default: on).
    pub fn interactive(mut self, on: bool) -> Self {
        self.interactive = on;
        self
    }

    /// Choose the fourth inset cell (default: the location map; pass
    /// [`CornerPanel::HazardType`] for the original watch box).
    pub fn corner_panel(mut self, corner: CornerPanel) -> Self {
        self.corner = corner;
        self
    }

    /// Skew-T title (top-left).
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Brand text drawn top-right above the hodograph column.
    pub fn brand(mut self, brand: impl Into<String>) -> Self {
        self.brand = Some(brand.into());
        self
    }

    pub fn parcel(mut self, parcel: ParcelType) -> Self {
        self.parcel = parcel;
        self
    }

    pub fn style(mut self, style: SkewTStyle) -> Self {
        self.style = style;
        self
    }

    pub fn size(mut self, size: Vec2) -> Self {
        self.size = Some(size);
        self
    }
}

fn weighted_horizontal_rects<const N: usize>(band: Rect, fractions: &[f32; N]) -> [Rect; N] {
    let mut fractions = *fractions;
    normalize_track_fractions(&mut fractions);
    let mut x = band.min.x;
    std::array::from_fn(|index| {
        let min = egui::pos2(x, band.min.y);
        x = if index == N - 1 {
            band.max.x
        } else {
            x + band.width() * fractions[index]
        };
        Rect::from_min_max(min, egui::pos2(x, band.max.y))
    })
}

fn bottom_active_columns(panels: &[PanelKind; 6]) -> [bool; 5] {
    [
        panels[0] != PanelKind::Hidden,
        panels[1] != PanelKind::Hidden,
        panels[2] != PanelKind::Hidden || panels[3] != PanelKind::Hidden,
        panels[4] != PanelKind::Hidden,
        panels[5] != PanelKind::Hidden,
    ]
}

fn weighted_bottom_rects(
    band: Rect,
    panels: &[PanelKind; 6],
    fractions: &[f32; 5],
    split_fraction: f32,
) -> [Rect; 6] {
    let mut base = *fractions;
    normalize_track_fractions(&mut base);
    let active = bottom_active_columns(panels);
    let fractions: [f32; 5] =
        std::array::from_fn(|index| if active[index] { base[index] } else { 0.0 });
    let total: f32 = fractions.iter().sum();
    let fractions = if total > 0.0 { fractions } else { base };
    let total: f32 = fractions.iter().sum();
    let mut x = band.min.x;
    let columns: [Rect; 5] = std::array::from_fn(|index| {
        let min = egui::pos2(x, band.min.y);
        x = if index == 4 {
            band.max.x
        } else {
            x + band.width() * fractions[index] / total
        };
        Rect::from_min_max(min, egui::pos2(x, band.max.y))
    });

    let collapsed = |at: egui::Pos2| Rect::from_min_max(at, at);
    let split = split_fraction.clamp(MIN_BOTTOM_SPLIT_FRACTION, MAX_BOTTOM_SPLIT_FRACTION);
    let middle_y = columns[2].min.y + columns[2].height() * split;
    let middle_top = Rect::from_min_max(columns[2].min, egui::pos2(columns[2].max.x, middle_y));
    let middle_bottom = Rect::from_min_max(egui::pos2(columns[2].min.x, middle_y), columns[2].max);
    let (slot2, slot3) = match (
        panels[2] != PanelKind::Hidden,
        panels[3] != PanelKind::Hidden,
    ) {
        (true, true) => (middle_top, middle_bottom),
        (true, false) => (columns[2], collapsed(columns[2].max)),
        (false, true) => (collapsed(columns[2].min), columns[2]),
        (false, false) => (collapsed(columns[2].min), collapsed(columns[2].min)),
    };
    [
        if active[0] {
            columns[0]
        } else {
            collapsed(columns[0].min)
        },
        if active[1] {
            columns[1]
        } else {
            collapsed(columns[1].min)
        },
        slot2,
        slot3,
        if active[3] {
            columns[3]
        } else {
            collapsed(columns[3].min)
        },
        if active[4] {
            columns[4]
        } else {
            collapsed(columns[4].min)
        },
    ]
}

/// Move one shared boundary while leaving every non-adjacent track unchanged.
/// `desired_fraction` is measured across the rendered allocation of `active`
/// tracks (hidden bottom tracks are omitted from that list).
fn set_active_track_boundary<const N: usize>(
    fractions: &mut [f32; N],
    active: &[usize],
    boundary: usize,
    desired_fraction: f32,
) {
    if boundary + 1 >= active.len() {
        return;
    }
    normalize_track_fractions(fractions);
    let total: f32 = active.iter().map(|index| fractions[*index]).sum();
    if total <= f32::EPSILON {
        return;
    }
    let left_index = active[boundary];
    let right_index = active[boundary + 1];
    let prefix: f32 = active[..boundary]
        .iter()
        .map(|index| fractions[*index])
        .sum();
    let pair = fractions[left_index] + fractions[right_index];
    let floor = (MIN_TRACK_FRACTION * total).min(pair * 0.49);
    let left = (desired_fraction.clamp(0.0, 1.0) * total - prefix).clamp(floor, pair - floor);
    fractions[left_index] = left;
    fractions[right_index] = pair - left;
}

fn vertical_resize_handle(
    ui: &mut Ui,
    painter: &egui::Painter,
    id: egui::Id,
    x: f32,
    top: f32,
    bottom: f32,
    tooltip: &'static str,
) -> Option<f32> {
    if bottom <= top {
        return None;
    }
    let hit = Rect::from_min_max(egui::pos2(x - 5.0, top), egui::pos2(x + 5.0, bottom));
    let response = ui
        .interact(hit, id, Sense::drag())
        .on_hover_cursor(egui::CursorIcon::ResizeHorizontal)
        .on_hover_text(tooltip);
    let color = if response.hovered() || response.dragged() {
        egui::Color32::WHITE
    } else {
        egui::Color32::from_rgb(0x04, 0xDB, 0xD8)
    };
    painter.line_segment(
        [egui::pos2(x, top), egui::pos2(x, bottom)],
        egui::Stroke::new(if response.dragged() { 2.0 } else { 1.0 }, color),
    );
    response
        .dragged()
        .then(|| response.interact_pointer_pos().map(|pos| pos.x))
        .flatten()
}

fn horizontal_resize_handle(
    ui: &mut Ui,
    painter: &egui::Painter,
    id: egui::Id,
    y: f32,
    left: f32,
    right: f32,
    tooltip: &'static str,
) -> Option<f32> {
    if right <= left {
        return None;
    }
    let hit = Rect::from_min_max(egui::pos2(left, y - 5.0), egui::pos2(right, y + 5.0));
    let response = ui
        .interact(hit, id, Sense::drag())
        .on_hover_cursor(egui::CursorIcon::ResizeVertical)
        .on_hover_text(tooltip);
    let color = if response.hovered() || response.dragged() {
        egui::Color32::WHITE
    } else {
        egui::Color32::from_rgb(0x04, 0xDB, 0xD8)
    };
    painter.line_segment(
        [egui::pos2(left, y), egui::pos2(right, y)],
        egui::Stroke::new(if response.dragged() { 2.0 } else { 1.0 }, color),
    );
    response
        .dragged()
        .then(|| response.interact_pointer_pos().map(|pos| pos.y))
        .flatten()
}

impl Widget for SoundingView<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        let size = self.size.unwrap_or_else(|| ui.available_size());
        let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
        if !ui.is_rect_visible(rect) || rect.width() < 200.0 || rect.height() < 150.0 {
            return response;
        }
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, self.style.bg_color);

        // --- Layout state (per-widget unless the host pinned an id via
        // `layout_memory_id`; edited in-app via the gear). ---
        let id = self
            .layout_id
            .unwrap_or_else(|| ui.id().with("sounding_layout"));
        let mut layout: SoundingLayout =
            ui.ctx().data_mut(|d| d.get_temp(id)).unwrap_or_else(|| {
                let mut l = SoundingLayout::default();
                if self.corner == CornerPanel::HazardType {
                    l.insets[3] = PanelKind::HazardType;
                }
                l
            });
        layout.normalize_geometry();

        let w = rect.width();
        let h = rect.height();

        // Shared track boundaries: every panel touching one of these reads the
        // exact same coordinate, so resizing cannot produce drifting borders.
        let band_top = rect.min.y + h * layout.top_height_fraction;
        let skew_right = rect.min.x + w * layout.skew_width_fraction;

        // --- Skew-T (its own Widget; place it in its cell). ---
        let skew_rect = Rect::from_min_max(rect.min, egui::pos2(skew_right, band_top));
        let mut skew_ui = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(skew_rect)
                .layout(egui::Layout::default()),
        );
        let mut skewt = SkewT::new(self.prof)
            .parcel(self.parcel)
            .title(self.title.clone())
            .style(self.style.clone())
            .cursor_readout(self.interactive)
            .interp_winds(self.interp_winds)
            .size(skew_rect.size());
        if let Some(on) = self.plot_omega {
            skewt = skewt.plot_omega(on);
        }
        skew_ui.add(skewt);

        // --- Upper right: brand band + grid2. ---
        let ur = Rect::from_min_max(
            egui::pos2(skew_right, rect.min.y),
            egui::pos2(rect.max.x, band_top),
        );
        let brand_h = 16.0f32;
        if let Some(brand) = &self.brand {
            // The model's name and start (e.g. "WOOF Hex · Global 120 km · Init GFS 22 Oct 2026 00Z"): bold so it
            // reads as the picture's identity, not a footnote.
            painter.text(
                egui::pos2(ur.max.x - 4.0, ur.min.y + 1.0),
                Align2::RIGHT_TOP,
                brand,
                self.style.bold_font(13.0),
                self.style.fg_color,
            );
        }
        let g2 = Rect::from_min_max(egui::pos2(ur.min.x, ur.min.y + brand_h), ur.max);
        let right_columns = weighted_horizontal_rects(g2, &layout.right_column_fractions);
        let strip_rects = [right_columns[0], right_columns[1]];
        let right_content = right_columns[2];
        let main_bottom =
            right_content.min.y + right_content.height() * layout.right_main_height_fraction;
        let main_rect = Rect::from_min_max(
            right_content.min,
            egui::pos2(right_content.max.x, main_bottom),
        );
        let inset_band = Rect::from_min_max(
            egui::pos2(right_content.min.x, main_bottom),
            right_content.max,
        );
        let inset_rects = weighted_horizontal_rects(inset_band, &layout.inset_column_fractions);

        // --- Bottom band cells. ---
        // Fully hidden columns surrender their allocation to visible columns,
        // so the optional final cell never leaves an empty quarter of the row.
        let band = Rect::from_min_max(egui::pos2(rect.min.x, band_top), rect.max);
        let bottom_rects = weighted_bottom_rects(
            band,
            &layout.bottom,
            &layout.bottom_column_fractions,
            layout.bottom_split_fraction,
        );

        // Scroll-to-zoom over the hodograph cell.
        if self.interactive
            && let Some(pos) = response.hover_pos()
            && main_rect.contains(pos)
            && layout.main == PanelKind::Hodograph
        {
            let scroll = ui.ctx().input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.0 {
                let factor = (-scroll as f64 / 400.0).exp();
                layout.hodo_zoom_kts = (layout.hodo_zoom_kts * factor).clamp(80.0, 500.0);
            }
        }

        let dv = self.derived;
        let st = &self.style;
        let zoom = layout.hodo_zoom_kts;
        for (kind, r) in layout
            .strips
            .iter()
            .zip(strip_rects.iter())
            .chain(layout.insets.iter().zip(inset_rects.iter()))
            .chain(layout.bottom.iter().zip(bottom_rects.iter()))
            .chain(std::iter::once((&layout.main, &main_rect)))
        {
            let configured_kind = match kind {
                PanelKind::ConvectiveIndices => Some(DiagnosticTablePanelKind::Convective),
                PanelKind::Kinematics => Some(DiagnosticTablePanelKind::Kinematics),
                PanelKind::SevereIndices => Some(DiagnosticTablePanelKind::Severe),
                _ => None,
            };
            let configured = self.diagnostic_tables.and_then(|tables| {
                configured_kind.and_then(|configured_kind| tables.panel(configured_kind))
            });
            if let Some(panel) = configured {
                crate::diagnostic_table::draw(&painter, *r, panel, st);
            } else if let Some(patches) = self.native_diagnostic_patches {
                match kind {
                    PanelKind::ConvectiveIndices
                        if patches.has_panel(DiagnosticTablePanelKind::Convective) =>
                    {
                        panels::index_board::draw_convective_patched(
                            &painter, *r, self.prof, dv, st, patches,
                        );
                    }
                    PanelKind::Kinematics
                        if patches.has_panel(DiagnosticTablePanelKind::Kinematics) =>
                    {
                        panels::index_board::draw_kinematics_patched(
                            &painter, *r, self.prof, dv, st, patches,
                        );
                    }
                    PanelKind::SevereIndices
                        if patches.has_panel(DiagnosticTablePanelKind::Severe) =>
                    {
                        panels::index_board::draw_indices_patched(
                            &painter, *r, self.prof, dv, st, patches,
                        );
                    }
                    PanelKind::IndexBoard if !patches.patches.is_empty() => {
                        panels::index_board::draw_patched(&painter, *r, self.prof, dv, st, patches);
                    }
                    _ => kind.draw(&painter, *r, self.prof, dv, st, zoom),
                }
            } else {
                kind.draw(&painter, *r, self.prof, dv, st, zoom);
            }
        }

        // Linked cursor: hovering the skew-T highlights the wind at that
        // height on the hodograph (wherever it currently lives).
        let hodo_cell = std::iter::once((&layout.main, &main_rect))
            .chain(layout.insets.iter().zip(inset_rects.iter()))
            .chain(layout.strips.iter().zip(strip_rects.iter()))
            .chain(layout.bottom.iter().zip(bottom_rects.iter()))
            .find(|(k, _)| **k == PanelKind::Hodograph)
            .map(|(_, r)| *r);
        if self.interactive
            && let Some(pos) = response.hover_pos()
            && skew_rect.contains(pos)
            && let Some(pres) = crate::skewt::hover_pressure(skew_rect, pos)
            && let Some(hodo_rect) = hodo_cell
        {
            let h_agl = self.prof.inner.to_agl(self.prof.inner.interp_hght(pres));
            if h_agl.is_finite() {
                panels::hodo::cursor_marker(&painter, hodo_rect, self.prof, st, h_agl, zoom);
            }
        }

        // --- Layout editor: gear button toggles per-cell pickers. ---
        if self.interactive {
            let edit_id = ui.id().with("sounding_layout_edit");
            let mut editing: bool = ui.ctx().data_mut(|d| d.get_temp(edit_id)).unwrap_or(false);
            let gear_rect = Rect::from_min_size(
                egui::pos2(rect.max.x - 24.0, band_top - 22.0),
                Vec2::new(22.0, 20.0),
            );
            if ui
                .put(gear_rect, egui::Button::new("\u{2699}").small())
                .on_hover_text("Edit panels and drag the cyan grid borders")
                .clicked()
            {
                editing = !editing;
            }
            if editing {
                {
                    let mut slots: Vec<(&mut PanelKind, Rect)> = Vec::new();
                    let SoundingLayout {
                        strips,
                        main,
                        insets,
                        bottom,
                        ..
                    } = &mut layout;
                    for (k, r) in strips.iter_mut().zip(strip_rects.iter()) {
                        slots.push((k, *r));
                    }
                    slots.push((main, main_rect));
                    for (k, r) in insets.iter_mut().zip(inset_rects.iter()) {
                        slots.push((k, *r));
                    }
                    for (k, r) in bottom.iter_mut().zip(bottom_rects.iter()) {
                        slots.push((k, *r));
                    }
                    for (i, (kind, r)) in slots.into_iter().enumerate() {
                        painter.rect_stroke(
                            r.shrink(1.0),
                            0.0,
                            egui::Stroke::new(1.0, egui::Color32::from_rgb(0x04, 0xDB, 0xD8)),
                            egui::StrokeKind::Inside,
                        );
                        if r.width() < 16.0 || r.height() < 22.0 {
                            continue;
                        }
                        let combo_rect = Rect::from_min_size(
                            r.min + Vec2::new(4.0, 4.0),
                            Vec2::new((r.width() - 8.0).min(150.0), 18.0),
                        );
                        let mut combo_ui = ui.new_child(
                            egui::UiBuilder::new()
                                .max_rect(combo_rect)
                                .layout(egui::Layout::default()),
                        );
                        egui::ComboBox::from_id_salt(id.with(i))
                            .selected_text(kind.label())
                            .width(combo_rect.width())
                            .show_ui(&mut combo_ui, |ui| {
                                for k in PanelKind::ALL {
                                    ui.selectable_value(kind, k, k.label());
                                }
                            });
                    }
                }

                let mut resized = false;
                if let Some(y) = horizontal_resize_handle(
                    ui,
                    &painter,
                    id.with("top_bottom_height"),
                    band_top,
                    rect.min.x,
                    rect.max.x,
                    "Drag to resize the upper and bottom sections",
                ) {
                    layout.top_height_fraction = ((y - rect.min.y) / h)
                        .clamp(MIN_TOP_HEIGHT_FRACTION, MAX_TOP_HEIGHT_FRACTION);
                    resized = true;
                }
                if let Some(x) = vertical_resize_handle(
                    ui,
                    &painter,
                    id.with("skew_right_width"),
                    skew_right,
                    rect.min.y,
                    band_top,
                    "Drag to resize the skew-T and right diagnostics",
                ) {
                    layout.skew_width_fraction = ((x - rect.min.x) / w)
                        .clamp(MIN_SKEW_WIDTH_FRACTION, MAX_SKEW_WIDTH_FRACTION);
                    resized = true;
                }
                if let Some(y) = horizontal_resize_handle(
                    ui,
                    &painter,
                    id.with("right_main_height"),
                    main_bottom,
                    right_content.min.x,
                    right_content.max.x,
                    "Drag to resize the large right panel and inset row",
                ) {
                    layout.right_main_height_fraction =
                        ((y - right_content.min.y) / right_content.height()).clamp(
                            MIN_RIGHT_MAIN_HEIGHT_FRACTION,
                            MAX_RIGHT_MAIN_HEIGHT_FRACTION,
                        );
                    resized = true;
                }

                let right_active = [0, 1, 2];
                for boundary in 0..2 {
                    if let Some(x) = vertical_resize_handle(
                        ui,
                        &painter,
                        id.with(("right_column", boundary)),
                        right_columns[boundary].max.x,
                        g2.min.y,
                        g2.max.y,
                        "Drag to resize the narrow strips and right panels",
                    ) {
                        set_active_track_boundary(
                            &mut layout.right_column_fractions,
                            &right_active,
                            boundary,
                            (x - g2.min.x) / g2.width(),
                        );
                        resized = true;
                    }
                }

                let inset_active = [0, 1, 2, 3];
                for boundary in 0..3 {
                    if let Some(x) = vertical_resize_handle(
                        ui,
                        &painter,
                        id.with(("inset_column", boundary)),
                        inset_rects[boundary].max.x,
                        inset_band.min.y,
                        inset_band.max.y,
                        "Drag to resize adjacent inset panels",
                    ) {
                        set_active_track_boundary(
                            &mut layout.inset_column_fractions,
                            &inset_active,
                            boundary,
                            (x - inset_band.min.x) / inset_band.width(),
                        );
                        resized = true;
                    }
                }

                let bottom_columns_active = bottom_active_columns(&layout.bottom);
                let bottom_active: Vec<usize> = bottom_columns_active
                    .iter()
                    .enumerate()
                    .filter_map(|(index, active)| (*active).then_some(index))
                    .collect();
                for (boundary, adjacent) in bottom_active.windows(2).enumerate() {
                    let left_rect = match adjacent[0] {
                        0 => bottom_rects[0],
                        1 => bottom_rects[1],
                        2 => {
                            if layout.bottom[2] != PanelKind::Hidden {
                                bottom_rects[2]
                            } else {
                                bottom_rects[3]
                            }
                        }
                        3 => bottom_rects[4],
                        4 => bottom_rects[5],
                        _ => unreachable!(),
                    };
                    if let Some(x) = vertical_resize_handle(
                        ui,
                        &painter,
                        id.with(("bottom_column", adjacent[0], adjacent[1])),
                        left_rect.max.x,
                        band.min.y,
                        band.max.y,
                        "Drag to resize adjacent bottom panels",
                    ) {
                        set_active_track_boundary(
                            &mut layout.bottom_column_fractions,
                            &bottom_active,
                            boundary,
                            (x - band.min.x) / band.width(),
                        );
                        resized = true;
                    }
                }
                if layout.bottom[2] != PanelKind::Hidden
                    && layout.bottom[3] != PanelKind::Hidden
                    && let Some(y) = horizontal_resize_handle(
                        ui,
                        &painter,
                        id.with("bottom_split_height"),
                        bottom_rects[2].max.y,
                        bottom_rects[2].min.x,
                        bottom_rects[2].max.x,
                        "Drag to resize the two stacked bottom panels",
                    )
                {
                    layout.bottom_split_fraction = ((y - band.min.y) / band.height())
                        .clamp(MIN_BOTTOM_SPLIT_FRACTION, MAX_BOTTOM_SPLIT_FRACTION);
                    resized = true;
                }
                if resized {
                    ui.ctx().request_repaint();
                }
                let reset_rect = Rect::from_min_size(
                    egui::pos2(rect.max.x - 84.0, band_top - 22.0),
                    Vec2::new(56.0, 20.0),
                );
                if ui
                    .put(reset_rect, egui::Button::new("reset").small())
                    .clicked()
                {
                    layout = SoundingLayout::default();
                }
            }
            ui.ctx().data_mut(|d| d.insert_temp(edit_id, editing));
        }
        ui.ctx().data_mut(|d| d.insert_temp(id, layout));

        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_panel_token_round_trips() {
        for k in PanelKind::ALL {
            assert_eq!(PanelKind::from_token(k.token()), Some(k), "{k:?}");
        }
        assert_eq!(PanelKind::from_token("nonsense"), None);
    }

    #[test]
    fn default_layout_round_trips_through_tokens() {
        let layout = SoundingLayout::default();
        let tokens = layout.to_tokens();
        assert_eq!(
            tokens,
            "speed,advection|hodograph|slinky,thetae,srwinds,locationmap|\
             convectiveindices,kinematics,ship,severeindices,streamwiseness,hidden|250"
        );
        assert_eq!(SoundingLayout::from_tokens(&tokens), Some(layout));
    }

    #[test]
    fn customized_layout_round_trips_through_tokens() {
        let mut layout = SoundingLayout::default();
        layout.strips[1] = PanelKind::Hidden;
        layout.main = PanelKind::Slinky;
        layout.insets[3] = PanelKind::HazardType;
        layout.bottom[5] = PanelKind::Stp;
        layout.hodo_zoom_kts = 137.5;
        assert_eq!(
            SoundingLayout::from_tokens(&layout.to_tokens()),
            Some(layout)
        );
    }

    #[test]
    fn customized_geometry_uses_versioned_tokens_and_round_trips() {
        let mut layout = SoundingLayout::default();
        layout.top_height_fraction = 0.58;
        layout.skew_width_fraction = 0.52;
        layout.right_main_height_fraction = 0.63;
        layout.right_column_fractions = [0.12, 0.10, 0.78];
        layout.inset_column_fractions = [0.20, 0.22, 0.28, 0.30];
        layout.bottom_column_fractions = [0.25, 0.20, 0.15, 0.18, 0.22];
        layout.bottom_split_fraction = 0.62;

        let tokens = layout.to_tokens();
        assert!(tokens.contains("|g2:"));
        assert_eq!(SoundingLayout::from_tokens(&tokens), Some(layout));
    }

    #[test]
    fn old_three_cell_bottom_tokens_expand_the_combined_index_board() {
        let layout = SoundingLayout::from_tokens(
            "speed,advection|hodograph|slinky,thetae,srwinds,locationmap|\
             indexboard,streamwiseness,stp|250",
        )
        .expect("legacy layout");
        assert_eq!(
            layout.bottom,
            [
                PanelKind::ConvectiveIndices,
                PanelKind::Kinematics,
                PanelKind::Ship,
                PanelKind::SevereIndices,
                PanelKind::Streamwiseness,
                PanelKind::Stp,
            ]
        );
        assert!(layout.has_default_geometry());
        assert!(!layout.to_tokens().contains("|g2:"));
    }

    #[test]
    fn legacy_g1_bottom_widths_migrate_without_changing_their_total_shares() {
        let layout = SoundingLayout::from_tokens(
            "speed,advection|hodograph|slinky,thetae,srwinds,locationmap|\
             indexboard,streamwiseness,stp|250|\
             g1:0.6,0.5,0.7;0.1,0.1,0.8;0.25,0.25,0.25,0.25;0.5,0.2,0.3",
        )
        .expect("legacy g1 layout");
        assert!((layout.bottom_column_fractions[0] - 0.19).abs() < 1.0e-6);
        assert!((layout.bottom_column_fractions[1] - 0.169).abs() < 1.0e-6);
        assert!((layout.bottom_column_fractions[2] - 0.141).abs() < 1.0e-6);
        assert!((layout.bottom_column_fractions[..3].iter().sum::<f32>() - 0.5).abs() < 1.0e-6);
        assert_eq!(layout.bottom_column_fractions[3], 0.2);
        assert_eq!(layout.bottom_column_fractions[4], 0.3);
        assert_eq!(layout.bottom_split_fraction, DEFAULT_BOTTOM_SPLIT_FRACTION);
    }

    #[test]
    fn g2_split_height_and_tracks_are_clamped() {
        let layout = SoundingLayout::from_tokens(
            "speed,advection|hodograph|slinky,thetae,srwinds,locationmap|\
             convectiveindices,kinematics,ship,severeindices,streamwiseness,hidden|250|\
             g2:0.6,0.5,0.7;0.1,0.1,0.8;0.25,0.25,0.25,0.25;0,0,0,0,1;9",
        )
        .expect("g2 geometry");
        assert_eq!(layout.bottom_split_fraction, MAX_BOTTOM_SPLIT_FRACTION);
        assert!(
            layout
                .bottom_column_fractions
                .iter()
                .all(|value| *value >= MIN_TRACK_FRACTION)
        );
        assert!((layout.bottom_column_fractions.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn from_tokens_tolerates_whitespace_and_clamps_zoom() {
        let layout = SoundingLayout::from_tokens(
            " speed , advection | hodograph | slinky,thetae,srwinds,hazardtype \
             | indexboard,streamwiseness,stp | 9000 ",
        )
        .expect("padded tokens parse");
        assert_eq!(layout.insets[3], PanelKind::HazardType);
        assert_eq!(layout.hodo_zoom_kts, 500.0, "zoom clamps to 80..=500");
    }

    #[test]
    fn from_tokens_rejects_malformed_input() {
        for bad in [
            "",
            "speed,advection|hodograph|slinky,thetae,srwinds,locationmap",
            "speed,advection|hodograph|slinky,thetae,srwinds,locationmap|indexboard,streamwiseness,stp|NaN",
            "speed,advection|hodograph|slinky,thetae,srwinds,locationmap|indexboard,streamwiseness,stp|250|extra",
            "speed,advection,speed|hodograph|slinky,thetae,srwinds,locationmap|indexboard,streamwiseness,stp|250",
            "speed,advection|hodograph|slinky,thetae,srwinds,teapot|indexboard,streamwiseness,stp|250",
            "speed,advection|hodograph|slinky,thetae,srwinds,locationmap|indexboard,streamwiseness,stp|250|g2:0.6,0.5,0.7;0.1,0.1,0.8;0.25,0.25,0.25,0.25;0.6,0.2,0.2",
            "speed,advection|hodograph|slinky,thetae,srwinds,locationmap|indexboard,streamwiseness,stp|250|g1:0.6,NaN,0.7;0.1,0.1,0.8;0.25,0.25,0.25,0.25;0.6,0.2,0.2",
            "speed,advection|hodograph|slinky,thetae,srwinds,locationmap|indexboard,streamwiseness,stp|250|g1:0.6,0.5,0.7;0.1,0.9;0.25,0.25,0.25,0.25;0.6,0.2,0.2",
        ] {
            assert_eq!(SoundingLayout::from_tokens(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn parsed_geometry_is_clamped_and_tracks_keep_a_minimum_share() {
        let layout = SoundingLayout::from_tokens(
            "speed,advection|hodograph|slinky,thetae,srwinds,locationmap|\
             indexboard,streamwiseness,stp|250|\
             g1:0.1,0.9,0.1;0,0,1;0,0,0,1;0,0,1",
        )
        .expect("geometry clamps");
        assert_eq!(layout.top_height_fraction, MIN_TOP_HEIGHT_FRACTION);
        assert_eq!(layout.skew_width_fraction, MAX_SKEW_WIDTH_FRACTION);
        assert_eq!(
            layout.right_main_height_fraction,
            MIN_RIGHT_MAIN_HEIGHT_FRACTION
        );
        for fractions in [
            layout.right_column_fractions.as_slice(),
            layout.inset_column_fractions.as_slice(),
            layout.bottom_column_fractions.as_slice(),
        ] {
            assert!(fractions.iter().all(|value| *value >= MIN_TRACK_FRACTION));
            assert!((fractions.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
        }
    }

    #[test]
    fn shared_track_drag_changes_only_the_adjacent_visible_tracks() {
        let mut fractions = [0.30, 0.10, 0.20, 0.15, 0.25];
        set_active_track_boundary(&mut fractions, &[0, 1, 3], 0, 0.5);
        assert!((fractions[0] - 0.275).abs() < 1.0e-6);
        assert!((fractions[1] - 0.125).abs() < 1.0e-6);
        assert_eq!(fractions[2], 0.20, "hidden track keeps its saved share");
        assert_eq!(fractions[3], 0.15, "non-adjacent track is unchanged");
        assert_eq!(fractions[4], 0.25, "hidden track keeps its saved share");
    }

    #[test]
    fn store_and_read_layout_via_pinned_id() {
        let ctx = egui::Context::default();
        let id = egui::Id::new("layout_memory_test");
        assert_eq!(stored_layout(&ctx, id), None);
        let mut layout = SoundingLayout::default();
        layout.hodo_zoom_kts = 210.0;
        store_layout(&ctx, id, &layout);
        assert_eq!(stored_layout(&ctx, id), Some(layout));
    }

    #[test]
    fn default_bottom_splits_the_index_board_and_reclaims_the_hidden_cell() {
        let layout = SoundingLayout::default();
        assert_eq!(
            layout.bottom,
            [
                PanelKind::ConvectiveIndices,
                PanelKind::Kinematics,
                PanelKind::Ship,
                PanelKind::SevereIndices,
                PanelKind::Streamwiseness,
                PanelKind::Hidden,
            ]
        );
        let band = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1000.0, 100.0));
        let rects = weighted_bottom_rects(
            band,
            &layout.bottom,
            &layout.bottom_column_fractions,
            layout.bottom_split_fraction,
        );
        assert!((rects[0].width() + rects[1].width() + rects[2].width() - 813.3333).abs() < 0.1);
        assert!((rects[4].width() - 186.6667).abs() < 0.1);
        assert_eq!(rects[5].width(), 0.0);
        assert!((rects[2].height() - 51.0).abs() < 0.1);
        assert!((rects[3].height() - 49.0).abs() < 0.1);
    }

    #[test]
    fn hiding_one_stacked_panel_gives_the_other_the_full_column() {
        let mut layout = SoundingLayout::default();
        layout.bottom[2] = PanelKind::Hidden;
        let band = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1000.0, 100.0));
        let rects = weighted_bottom_rects(
            band,
            &layout.bottom,
            &layout.bottom_column_fractions,
            layout.bottom_split_fraction,
        );
        assert_eq!(rects[2].size(), Vec2::ZERO);
        assert_eq!(rects[3].height(), band.height());
    }
}
