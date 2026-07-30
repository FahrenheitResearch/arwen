use crate::color::Rgba;
use crate::colormap::{LevelDensity, RenderDensity};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductVisualMode {
    FilledMeteorology,
    UpperAirAnalysis,
    OverlayAnalysis,
    SevereDiagnostic,
    PanelMember,
    ComparisonPanel,
}

impl Default for ProductVisualMode {
    fn default() -> Self {
        Self::FilledMeteorology
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StaticPlotStyle {
    #[default]
    Default,
    CleanAtlas,
    CleanAtlasFast,
    CleanAtlasQuality2x,
    CleanAtlasCombined,
    Operational,
    OperationalFast,
    OperationalQuality2x,
    OperationalBudget30s,
}

impl StaticPlotStyle {
    pub fn from_env() -> Self {
        std::env::var("RUSTWX_PLOT_STYLE")
            .ok()
            .and_then(|value| Self::parse(&value))
            .unwrap_or(Self::OperationalFast)
    }

    pub fn parse(value: &str) -> Option<Self> {
        let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
        match normalized.as_str() {
            "" | "default" | "classic" | "baseline" | "standard" => Some(Self::Default),
            "clean" | "atlas" | "clean_atlas" | "pivotal" => Some(Self::CleanAtlas),
            "fast" | "clean_fast" | "atlas_fast" | "clean_atlas_fast" | "production" => {
                Some(Self::CleanAtlasFast)
            }
            "quality"
            | "quality_2x"
            | "beauty"
            | "export"
            | "clean_quality"
            | "clean_quality_2x"
            | "clean_atlas_quality"
            | "clean_atlas_quality_2x" => Some(Self::CleanAtlasQuality2x),
            "combined"
            | "clean_combined"
            | "atlas_combined"
            | "clean_atlas_combined"
            | "presentation"
            | "best" => Some(Self::CleanAtlasCombined),
            "operational" | "ops" | "weathermodels" | "reference" => Some(Self::Operational),
            "operational_fast" | "ops_fast" | "weathermodels_fast" | "pivotal_fast" => {
                Some(Self::OperationalFast)
            }
            "operational_quality"
            | "operational_quality_2x"
            | "ops_quality"
            | "weathermodels_quality"
            | "pivotal_quality" => Some(Self::OperationalQuality2x),
            "operational_budget_30s"
            | "operational_budget"
            | "ops_budget"
            | "budget_30s"
            | "quality_budget"
            | "operational_best"
            | "ops_best"
            | "weathermodels_best"
            | "max_quality" => Some(Self::OperationalBudget30s),
            _ => None,
        }
    }

    pub fn uses_clean_atlas_presentation(self) -> bool {
        matches!(
            self,
            Self::CleanAtlas
                | Self::CleanAtlasFast
                | Self::CleanAtlasQuality2x
                | Self::CleanAtlasCombined
                | Self::Operational
                | Self::OperationalFast
                | Self::OperationalQuality2x
                | Self::OperationalBudget30s
        )
    }

    pub fn uses_operational_presentation(self) -> bool {
        matches!(
            self,
            Self::Operational
                | Self::OperationalFast
                | Self::OperationalQuality2x
                | Self::OperationalBudget30s
        )
    }

    pub fn render_density(self, requested: RenderDensity) -> RenderDensity {
        if !matches!(
            self,
            Self::CleanAtlasFast
                | Self::CleanAtlasQuality2x
                | Self::CleanAtlasCombined
                | Self::OperationalFast
                | Self::OperationalQuality2x
                | Self::OperationalBudget30s
        ) {
            return requested;
        }

        RenderDensity {
            fill: LevelDensity {
                multiplier: requested.fill.multiplier.max(16),
                min_source_level_count: requested.fill.min_source_level_count.min(5),
            },
            palette_multiplier: requested.palette_multiplier.max(16),
        }
    }

    pub fn supersample_factor(self, requested: u32) -> u32 {
        match self {
            Self::OperationalBudget30s => requested.max(3),
            Self::CleanAtlasQuality2x | Self::CleanAtlasCombined | Self::OperationalQuality2x => {
                requested.max(2)
            }
            _ => requested.max(1),
        }
    }

    pub fn supersample_sharpen(self, requested: bool) -> bool {
        match self {
            Self::CleanAtlasQuality2x | Self::OperationalQuality2x => false,
            Self::CleanAtlasCombined | Self::OperationalBudget30s => true,
            _ => requested,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LineworkRole {
    Coast,
    Lake,
    International,
    State,
    County,
    Generic,
}

impl Default for LineworkRole {
    fn default() -> Self {
        Self::Generic
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolygonRole {
    Ocean,
    Land,
    Lake,
    Generic,
}

impl Default for PolygonRole {
    fn default() -> Self {
        Self::Generic
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TitleAnchor {
    Center,
    Left,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorbarOrientation {
    HorizontalBottom,
    VerticalRight,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutMetrics {
    pub margin_x: u32,
    pub title_h: u32,
    pub footer_h: u32,
    pub colorbar_h: u32,
    pub colorbar_gap: u32,
    pub colorbar_margin_x: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineworkStyle {
    pub visible: bool,
    pub color: Rgba,
    pub width: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolygonStyle {
    pub visible: bool,
    pub color: Rgba,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChromeStyle {
    pub title_anchor: TitleAnchor,
    pub title_color: Rgba,
    pub subtitle_color: Rgba,
    pub frame_color: Option<Rgba>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorbarPresentation {
    pub orientation: ColorbarOrientation,
    pub frame_color: Rgba,
    pub divider_color: Rgba,
    pub tick_color: Rgba,
    pub label_color: Rgba,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderPresentation {
    pub mode: ProductVisualMode,
    pub plot_style: StaticPlotStyle,
    pub canvas_background: Rgba,
    pub map_background: Rgba,
    pub domain_boundary: Option<LineworkStyle>,
    pub chrome: ChromeStyle,
    pub colorbar: ColorbarPresentation,
    pub layout: LayoutMetrics,
}

impl RenderPresentation {
    pub fn for_mode(mode: ProductVisualMode) -> Self {
        Self::for_mode_with_style(mode, StaticPlotStyle::Default)
    }

    pub fn for_mode_from_env(mode: ProductVisualMode) -> Self {
        Self::for_mode_with_style(mode, StaticPlotStyle::from_env())
    }

    pub fn for_mode_with_style(mode: ProductVisualMode, plot_style: StaticPlotStyle) -> Self {
        let mut presentation = match mode {
            ProductVisualMode::FilledMeteorology => filled_meteorology(),
            ProductVisualMode::UpperAirAnalysis => upper_air_analysis(),
            ProductVisualMode::OverlayAnalysis => overlay_analysis(),
            ProductVisualMode::SevereDiagnostic => severe_diagnostic(),
            ProductVisualMode::PanelMember => panel_member(),
            ProductVisualMode::ComparisonPanel => comparison_panel(),
        };
        presentation.plot_style = plot_style;
        presentation.apply_static_plot_style();
        presentation
    }

    fn apply_static_plot_style(&mut self) {
        if !self.plot_style.uses_clean_atlas_presentation() {
            return;
        }
        self.apply_clean_atlas_plot_style();
        if self.plot_style.uses_operational_presentation() {
            self.apply_operational_plot_style();
        }
    }

    fn apply_clean_atlas_plot_style(&mut self) {
        self.canvas_background = Rgba::new(246, 248, 250);
        self.chrome = clean_atlas_chrome(self.chrome.title_anchor);
        self.colorbar = clean_atlas_colorbar();
        self.layout = if matches!(
            self.mode,
            ProductVisualMode::PanelMember | ProductVisualMode::ComparisonPanel
        ) {
            clean_atlas_compact_layout()
        } else {
            clean_atlas_layout()
        };

        match self.mode {
            ProductVisualMode::UpperAirAnalysis => {
                self.map_background = Rgba::new(242, 244, 243);
            }
            ProductVisualMode::OverlayAnalysis => {
                self.map_background = Rgba::new(250, 251, 252);
            }
            ProductVisualMode::SevereDiagnostic => {
                self.map_background = Rgba::new(251, 252, 251);
            }
            ProductVisualMode::FilledMeteorology
            | ProductVisualMode::PanelMember
            | ProductVisualMode::ComparisonPanel => {
                self.map_background = Rgba::new(249, 250, 248);
            }
        }
    }

    fn apply_operational_plot_style(&mut self) {
        self.colorbar = operational_colorbar();
        self.layout.colorbar_gap = self.layout.colorbar_gap.min(6);
        self.layout.colorbar_margin_x = self.layout.colorbar_margin_x.min(72);
        if matches!(
            self.mode,
            ProductVisualMode::PanelMember | ProductVisualMode::ComparisonPanel
        ) {
            self.colorbar.orientation = ColorbarOrientation::HorizontalBottom;
        }
    }

    pub fn polygon_style(self, role: PolygonRole, fallback: Rgba) -> PolygonStyle {
        if self.plot_style.uses_clean_atlas_presentation() {
            return clean_atlas_polygon_style(self.mode, role, fallback);
        }

        match self.mode {
            ProductVisualMode::OverlayAnalysis => match role {
                PolygonRole::Ocean => PolygonStyle {
                    visible: true,
                    color: Rgba::new(247, 250, 253),
                },
                PolygonRole::Land => PolygonStyle {
                    visible: true,
                    color: Rgba::new(255, 255, 255),
                },
                PolygonRole::Lake => PolygonStyle {
                    visible: true,
                    color: Rgba::new(240, 247, 252),
                },
                PolygonRole::Generic => PolygonStyle {
                    visible: true,
                    color: fallback,
                },
            },
            ProductVisualMode::UpperAirAnalysis => match role {
                PolygonRole::Ocean => PolygonStyle {
                    visible: true,
                    color: Rgba::new(242, 246, 250),
                },
                PolygonRole::Land => PolygonStyle {
                    visible: true,
                    // Upper-air products can legitimately mask below-ground
                    // isobaric surfaces over higher terrain. Use a faint
                    // terrain-tinted land fill so those regions do not read as
                    // a rendering hole.
                    color: Rgba::new(232, 228, 217),
                },
                PolygonRole::Lake => PolygonStyle {
                    visible: true,
                    color: Rgba::new(232, 241, 247),
                },
                PolygonRole::Generic => PolygonStyle {
                    visible: true,
                    color: fallback,
                },
            },
            ProductVisualMode::SevereDiagnostic => match role {
                PolygonRole::Ocean => PolygonStyle {
                    visible: true,
                    color: Rgba::new(246, 250, 253),
                },
                PolygonRole::Land => PolygonStyle {
                    visible: true,
                    color: Rgba::new(252, 252, 249),
                },
                PolygonRole::Lake => PolygonStyle {
                    visible: true,
                    color: Rgba::new(239, 246, 251),
                },
                PolygonRole::Generic => PolygonStyle {
                    visible: true,
                    color: fallback,
                },
            },
            ProductVisualMode::PanelMember | ProductVisualMode::ComparisonPanel => match role {
                PolygonRole::Ocean => PolygonStyle {
                    visible: false,
                    color: Rgba::TRANSPARENT,
                },
                PolygonRole::Land => PolygonStyle {
                    visible: false,
                    color: Rgba::TRANSPARENT,
                },
                PolygonRole::Lake => PolygonStyle {
                    visible: true,
                    color: Rgba::new(242, 247, 250),
                },
                PolygonRole::Generic => PolygonStyle {
                    visible: true,
                    color: fallback,
                },
            },
            ProductVisualMode::FilledMeteorology => match role {
                PolygonRole::Ocean => PolygonStyle {
                    visible: false,
                    color: Rgba::TRANSPARENT,
                },
                PolygonRole::Land => PolygonStyle {
                    visible: false,
                    color: Rgba::TRANSPARENT,
                },
                PolygonRole::Lake => PolygonStyle {
                    visible: true,
                    color: Rgba::new(242, 247, 250),
                },
                PolygonRole::Generic => PolygonStyle {
                    visible: true,
                    color: fallback,
                },
            },
        }
    }

    pub fn linework_style(
        self,
        role: LineworkRole,
        fallback: Rgba,
        fallback_width: u32,
    ) -> LineworkStyle {
        if self.plot_style.uses_clean_atlas_presentation() {
            return clean_atlas_linework_style(self.mode, role, fallback, fallback_width);
        }

        let (color, width, visible) = match self.mode {
            ProductVisualMode::OverlayAnalysis => match role {
                LineworkRole::Coast => (Rgba::with_alpha(24, 28, 34, 210), 2, true),
                LineworkRole::Lake => (Rgba::with_alpha(60, 66, 74, 160), 2, true),
                LineworkRole::International => (Rgba::new(74, 82, 94), 1, true),
                LineworkRole::State => (Rgba::with_alpha(24, 28, 34, 210), 2, true),
                LineworkRole::County => (Rgba::with_alpha(142, 151, 162, 150), 1, true),
                LineworkRole::Generic => (fallback, fallback_width.max(1), true),
            },
            ProductVisualMode::UpperAirAnalysis => match role {
                LineworkRole::Coast => (Rgba::with_alpha(22, 26, 32, 220), 2, true),
                LineworkRole::Lake => (Rgba::with_alpha(58, 64, 72, 165), 2, true),
                LineworkRole::International => (Rgba::new(68, 76, 86), 1, true),
                LineworkRole::State => (Rgba::with_alpha(22, 26, 32, 220), 2, true),
                LineworkRole::County => (Rgba::with_alpha(150, 158, 168, 90), 1, true),
                LineworkRole::Generic => (fallback, fallback_width.max(1), true),
            },
            ProductVisualMode::SevereDiagnostic => match role {
                LineworkRole::Coast => (Rgba::with_alpha(38, 46, 56, 185), 1, true),
                LineworkRole::Lake => (Rgba::with_alpha(82, 96, 110, 125), 1, true),
                LineworkRole::International => (Rgba::with_alpha(72, 82, 94, 170), 1, true),
                LineworkRole::State => (Rgba::with_alpha(46, 54, 64, 170), 1, true),
                LineworkRole::County => (Rgba::with_alpha(126, 134, 143, 90), 1, false),
                LineworkRole::Generic => (fallback, fallback_width.max(1), true),
            },
            ProductVisualMode::PanelMember | ProductVisualMode::ComparisonPanel => match role {
                LineworkRole::Coast => (Rgba::with_alpha(26, 30, 36, 215), 2, true),
                LineworkRole::Lake => (Rgba::with_alpha(68, 76, 86, 145), 2, true),
                LineworkRole::International => (Rgba::new(92, 100, 110), 1, true),
                LineworkRole::State => (Rgba::with_alpha(26, 30, 36, 215), 2, true),
                LineworkRole::County => (Rgba::with_alpha(150, 158, 168, 70), 1, true),
                LineworkRole::Generic => (fallback, fallback_width.max(1), true),
            },
            ProductVisualMode::FilledMeteorology => match role {
                LineworkRole::Coast => (Rgba::with_alpha(22, 26, 32, 220), 2, true),
                LineworkRole::Lake => (Rgba::with_alpha(54, 60, 68, 150), 2, true),
                LineworkRole::International => (Rgba::with_alpha(72, 80, 92, 210), 1, true),
                LineworkRole::State => (Rgba::with_alpha(22, 26, 32, 220), 2, true),
                LineworkRole::County => (Rgba::with_alpha(140, 148, 160, 70), 1, false),
                LineworkRole::Generic => (fallback, fallback_width.max(1), true),
            },
        };
        LineworkStyle {
            visible,
            color,
            width,
        }
    }

    pub fn domain_frame_style(self, requested: Rgba, requested_width: u32) -> LineworkStyle {
        if !self.plot_style.uses_clean_atlas_presentation() {
            return LineworkStyle {
                visible: true,
                color: requested,
                width: requested_width.max(1),
            };
        }
        let width = if requested_width <= 4 {
            requested_width.max(1).min(2)
        } else {
            requested_width
        };
        LineworkStyle {
            visible: true,
            color: Rgba::with_alpha(18, 24, 32, 235),
            width,
        }
    }

    pub fn contour_color(self, requested: Rgba) -> Rgba {
        match self.mode {
            ProductVisualMode::UpperAirAnalysis | ProductVisualMode::OverlayAnalysis => {
                Rgba::new(30, 36, 44)
            }
            ProductVisualMode::SevereDiagnostic => Rgba::new(36, 38, 40),
            ProductVisualMode::PanelMember | ProductVisualMode::ComparisonPanel => {
                if requested == Rgba::BLACK {
                    Rgba::new(42, 48, 56)
                } else {
                    requested
                }
            }
            ProductVisualMode::FilledMeteorology => requested,
        }
    }

    pub fn barb_color(self, requested: Rgba) -> Rgba {
        match self.mode {
            ProductVisualMode::UpperAirAnalysis | ProductVisualMode::OverlayAnalysis => {
                Rgba::new(28, 34, 42)
            }
            ProductVisualMode::SevereDiagnostic => Rgba::new(34, 38, 42),
            ProductVisualMode::PanelMember | ProductVisualMode::ComparisonPanel => {
                Rgba::new(44, 50, 58)
            }
            ProductVisualMode::FilledMeteorology => requested,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_plot_style_parses_clean_atlas_aliases() {
        assert_eq!(
            StaticPlotStyle::parse("clean_atlas"),
            Some(StaticPlotStyle::CleanAtlas)
        );
        assert_eq!(
            StaticPlotStyle::parse("operational"),
            Some(StaticPlotStyle::Operational)
        );
        assert_eq!(
            StaticPlotStyle::parse("pivotal"),
            Some(StaticPlotStyle::CleanAtlas)
        );
        assert_eq!(
            StaticPlotStyle::parse("operational_fast"),
            Some(StaticPlotStyle::OperationalFast)
        );
        assert_eq!(
            StaticPlotStyle::parse("clean_atlas_fast"),
            Some(StaticPlotStyle::CleanAtlasFast)
        );
        assert_eq!(
            StaticPlotStyle::parse("clean_atlas_quality_2x"),
            Some(StaticPlotStyle::CleanAtlasQuality2x)
        );
        assert_eq!(
            StaticPlotStyle::parse("clean_atlas_combined"),
            Some(StaticPlotStyle::CleanAtlasCombined)
        );
        assert_eq!(
            StaticPlotStyle::parse("operational_budget_30s"),
            Some(StaticPlotStyle::OperationalBudget30s)
        );
        assert_eq!(
            StaticPlotStyle::parse("max_quality"),
            Some(StaticPlotStyle::OperationalBudget30s)
        );
        assert_eq!(
            StaticPlotStyle::parse("default"),
            Some(StaticPlotStyle::Default)
        );
        assert_eq!(StaticPlotStyle::parse("unknown"), None);
    }

    #[test]
    fn clean_atlas_uses_broader_chrome_spacing() {
        let default = RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology);
        let clean = RenderPresentation::for_mode_with_style(
            ProductVisualMode::FilledMeteorology,
            StaticPlotStyle::CleanAtlas,
        );

        assert_eq!(clean.plot_style, StaticPlotStyle::CleanAtlas);
        assert!(clean.layout.title_h > default.layout.title_h);
        assert!(clean.layout.footer_h > default.layout.footer_h);
        assert!(clean.chrome.frame_color.is_some());
    }

    #[test]
    fn clean_atlas_fast_uses_clean_atlas_presentation_and_dense_rendering() {
        let clean = RenderPresentation::for_mode_with_style(
            ProductVisualMode::FilledMeteorology,
            StaticPlotStyle::CleanAtlasFast,
        );
        let density = StaticPlotStyle::CleanAtlasFast.render_density(RenderDensity {
            fill: LevelDensity::default(),
            palette_multiplier: 1,
        });

        assert_eq!(clean.plot_style, StaticPlotStyle::CleanAtlasFast);
        assert!(clean.chrome.frame_color.is_some());
        assert_eq!(density.fill.multiplier, 16);
        assert_eq!(density.fill.min_source_level_count, 5);
        assert_eq!(density.palette_multiplier, 16);
    }

    #[test]
    fn quality_styles_are_first_class_supersample_modes() {
        assert_eq!(
            StaticPlotStyle::CleanAtlasQuality2x.supersample_factor(1),
            2
        );
        assert_eq!(StaticPlotStyle::CleanAtlasCombined.supersample_factor(1), 2);
        assert_eq!(
            StaticPlotStyle::OperationalBudget30s.supersample_factor(1),
            3
        );
        assert!(!StaticPlotStyle::CleanAtlasQuality2x.supersample_sharpen(true));
        assert!(StaticPlotStyle::CleanAtlasCombined.supersample_sharpen(false));
        assert!(StaticPlotStyle::OperationalBudget30s.supersample_sharpen(false));
    }

    #[test]
    fn clean_atlas_uses_sleek_basemap_line_widths() {
        let style = RenderPresentation::for_mode_with_style(
            ProductVisualMode::FilledMeteorology,
            StaticPlotStyle::CleanAtlas,
        )
        .linework_style(LineworkRole::Coast, Rgba::BLACK, 1);

        assert!(style.visible);
        assert_eq!(style.width, 1);
    }

    #[test]
    fn operational_style_uses_crisp_black_chrome_and_linework() {
        let presentation = RenderPresentation::for_mode_with_style(
            ProductVisualMode::FilledMeteorology,
            StaticPlotStyle::OperationalFast,
        );
        let coast = presentation.linework_style(LineworkRole::Coast, Rgba::BLACK, 1);
        let land = presentation.polygon_style(PolygonRole::Land, Rgba::new(250, 250, 246));

        assert_eq!(presentation.canvas_background, Rgba::new(246, 248, 250));
        assert_eq!(
            presentation.chrome.frame_color,
            Some(Rgba::with_alpha(46, 56, 68, 118))
        );
        assert_eq!(
            presentation.colorbar.orientation,
            ColorbarOrientation::VerticalRight
        );
        assert_eq!(presentation.colorbar.frame_color, Rgba::BLACK);
        assert_eq!(coast.width, 1);
        assert_eq!(coast.color, Rgba::with_alpha(18, 22, 28, 227));
        assert!(land.visible);
        assert_eq!(land.color, Rgba::new(250, 250, 246));
    }

    #[test]
    fn operational_filled_political_linework_uses_wrf_hardened_clean_atlas_values() {
        let presentation = RenderPresentation::for_mode_with_style(
            ProductVisualMode::FilledMeteorology,
            StaticPlotStyle::OperationalFast,
        );
        let international =
            presentation.linework_style(LineworkRole::International, Rgba::BLACK, 1);
        let state = presentation.linework_style(LineworkRole::State, Rgba::BLACK, 1);

        assert_eq!(international.width, 1);
        assert_eq!(international.color, Rgba::with_alpha(30, 36, 44, 209));
        assert_eq!(state.width, 1);
        assert_eq!(state.color, Rgba::with_alpha(18, 22, 28, 220));
    }

    #[test]
    fn operational_filled_maps_show_faint_county_linework() {
        let presentation = RenderPresentation::for_mode_with_style(
            ProductVisualMode::FilledMeteorology,
            StaticPlotStyle::OperationalFast,
        );
        let county = presentation.linework_style(LineworkRole::County, Rgba::BLACK, 1);

        assert!(county.visible);
        assert_eq!(county.width, 1);
        assert_eq!(county.color, Rgba::with_alpha(92, 100, 112, 55));
    }

    #[test]
    fn operational_upper_air_uses_wrf_hardened_clean_atlas_admin_linework() {
        let presentation = RenderPresentation::for_mode_with_style(
            ProductVisualMode::UpperAirAnalysis,
            StaticPlotStyle::OperationalFast,
        );
        let county = presentation.linework_style(LineworkRole::County, Rgba::BLACK, 1);
        let state = presentation.linework_style(LineworkRole::State, Rgba::BLACK, 2);

        assert!(!county.visible);
        assert_eq!(county.color, Rgba::with_alpha(92, 100, 112, 55));
        assert_eq!(state.width, 2);
        assert_eq!(state.color, Rgba::with_alpha(18, 22, 28, 220));
    }

    #[test]
    fn filled_meteorology_keeps_lake_linework_visible() {
        let style = RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology)
            .linework_style(LineworkRole::Lake, Rgba::BLACK, 3);

        assert!(style.visible);
        assert_eq!(style.width, 2);
        assert_eq!(style.color, Rgba::with_alpha(54, 60, 68, 150));
    }

    #[test]
    fn filled_meteorology_uses_dark_thicker_state_lines() {
        let style = RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology)
            .linework_style(LineworkRole::State, Rgba::BLACK, 1);

        assert!(style.visible);
        assert_eq!(style.width, 2);
        assert_eq!(style.color, Rgba::with_alpha(22, 26, 32, 220));
    }

    #[test]
    fn filled_meteorology_aligns_coast_and_state_lines() {
        let style = RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology)
            .linework_style(LineworkRole::Coast, Rgba::BLACK, 1);

        assert!(style.visible);
        assert_eq!(style.width, 2);
        assert_eq!(style.color, Rgba::with_alpha(22, 26, 32, 220));
    }

    #[test]
    fn clean_atlas_lake_linework_is_neutral_not_blue() {
        let style = RenderPresentation::for_mode_with_style(
            ProductVisualMode::FilledMeteorology,
            StaticPlotStyle::CleanAtlasFast,
        )
        .linework_style(LineworkRole::Lake, Rgba::BLACK, 1);

        assert!(style.visible);
        assert_eq!(style.color, Rgba::with_alpha(64, 70, 78, 125));
    }
}

fn common_chrome(title_anchor: TitleAnchor, frame_color: Option<Rgba>) -> ChromeStyle {
    ChromeStyle {
        title_anchor,
        title_color: Rgba::BLACK,
        subtitle_color: Rgba::BLACK,
        frame_color,
    }
}

fn common_colorbar() -> ColorbarPresentation {
    ColorbarPresentation {
        orientation: ColorbarOrientation::HorizontalBottom,
        frame_color: Rgba::new(92, 100, 112),
        divider_color: Rgba::with_alpha(255, 255, 255, 70),
        tick_color: Rgba::new(92, 100, 112),
        label_color: Rgba::BLACK,
    }
}

fn normal_layout() -> LayoutMetrics {
    LayoutMetrics {
        margin_x: 18,
        title_h: 42,
        footer_h: 30,
        colorbar_h: 12,
        colorbar_gap: 8,
        colorbar_margin_x: 86,
    }
}

fn compact_layout() -> LayoutMetrics {
    LayoutMetrics {
        margin_x: 8,
        title_h: 34,
        footer_h: 24,
        colorbar_h: 10,
        colorbar_gap: 8,
        colorbar_margin_x: 42,
    }
}

fn filled_meteorology() -> RenderPresentation {
    RenderPresentation {
        mode: ProductVisualMode::FilledMeteorology,
        plot_style: StaticPlotStyle::Default,
        canvas_background: Rgba::new(247, 248, 250),
        map_background: Rgba::new(250, 250, 247),
        domain_boundary: None,
        chrome: common_chrome(TitleAnchor::Left, None),
        colorbar: common_colorbar(),
        layout: normal_layout(),
    }
}

fn upper_air_analysis() -> RenderPresentation {
    RenderPresentation {
        mode: ProductVisualMode::UpperAirAnalysis,
        plot_style: StaticPlotStyle::Default,
        canvas_background: Rgba::new(246, 247, 249),
        map_background: Rgba::new(238, 235, 227),
        domain_boundary: None,
        chrome: common_chrome(TitleAnchor::Left, None),
        colorbar: common_colorbar(),
        layout: normal_layout(),
    }
}

fn overlay_analysis() -> RenderPresentation {
    RenderPresentation {
        mode: ProductVisualMode::OverlayAnalysis,
        plot_style: StaticPlotStyle::Default,
        canvas_background: Rgba::WHITE,
        map_background: Rgba::WHITE,
        domain_boundary: None,
        chrome: common_chrome(TitleAnchor::Left, None),
        colorbar: common_colorbar(),
        layout: normal_layout(),
    }
}

fn severe_diagnostic() -> RenderPresentation {
    RenderPresentation {
        mode: ProductVisualMode::SevereDiagnostic,
        plot_style: StaticPlotStyle::Default,
        canvas_background: Rgba::new(247, 248, 249),
        map_background: Rgba::new(252, 253, 251),
        domain_boundary: None,
        chrome: common_chrome(TitleAnchor::Left, None),
        colorbar: common_colorbar(),
        layout: normal_layout(),
    }
}

fn panel_member() -> RenderPresentation {
    RenderPresentation {
        mode: ProductVisualMode::PanelMember,
        plot_style: StaticPlotStyle::Default,
        canvas_background: Rgba::new(246, 247, 249),
        map_background: Rgba::new(250, 250, 247),
        domain_boundary: None,
        chrome: common_chrome(TitleAnchor::Left, None),
        colorbar: common_colorbar(),
        layout: compact_layout(),
    }
}

fn comparison_panel() -> RenderPresentation {
    let mut presentation = panel_member();
    presentation.mode = ProductVisualMode::ComparisonPanel;
    presentation
}

fn clean_atlas_polygon_style(
    mode: ProductVisualMode,
    role: PolygonRole,
    fallback: Rgba,
) -> PolygonStyle {
    match mode {
        ProductVisualMode::OverlayAnalysis => match role {
            PolygonRole::Ocean => PolygonStyle {
                visible: true,
                color: Rgba::new(239, 245, 249),
            },
            PolygonRole::Land => PolygonStyle {
                visible: true,
                color: Rgba::new(250, 250, 246),
            },
            PolygonRole::Lake => PolygonStyle {
                visible: true,
                color: Rgba::new(231, 240, 247),
            },
            PolygonRole::Generic => PolygonStyle {
                visible: true,
                color: fallback,
            },
        },
        ProductVisualMode::UpperAirAnalysis => match role {
            PolygonRole::Ocean => PolygonStyle {
                visible: true,
                color: Rgba::new(238, 243, 247),
            },
            PolygonRole::Land => PolygonStyle {
                visible: true,
                color: Rgba::new(245, 245, 240),
            },
            PolygonRole::Lake => PolygonStyle {
                visible: true,
                color: Rgba::new(228, 238, 246),
            },
            PolygonRole::Generic => PolygonStyle {
                visible: true,
                color: fallback,
            },
        },
        ProductVisualMode::SevereDiagnostic
        | ProductVisualMode::PanelMember
        | ProductVisualMode::ComparisonPanel
        | ProductVisualMode::FilledMeteorology => match role {
            PolygonRole::Ocean => PolygonStyle {
                visible: true,
                color: Rgba::new(238, 244, 249),
            },
            PolygonRole::Land => PolygonStyle {
                visible: true,
                color: Rgba::new(250, 250, 246),
            },
            PolygonRole::Lake => PolygonStyle {
                visible: true,
                color: Rgba::new(229, 239, 247),
            },
            PolygonRole::Generic => PolygonStyle {
                visible: true,
                color: fallback,
            },
        },
    }
}

fn clean_atlas_linework_style(
    mode: ProductVisualMode,
    role: LineworkRole,
    fallback: Rgba,
    fallback_width: u32,
) -> LineworkStyle {
    let fallback_width = fallback_width.max(1);
    let major_width = fallback_width.clamp(1, 2);
    let minor_width = fallback_width.clamp(1, 2);
    let county_visible = clean_atlas_county_linework_visible(mode);
    let width_boost = static_linework_width_boost();
    let alpha_scale = static_linework_alpha_scale();
    let boost_width = |width: u32| width.saturating_add(width_boost).clamp(1, 8);
    let alpha = |cap: u8| {
        let base = fallback.a.min(cap) as f32;
        (base * alpha_scale).round().clamp(0.0, 255.0) as u8
    };
    let (color, width, visible) = match role {
        LineworkRole::Coast => (
            Rgba::with_alpha(18, 22, 28, alpha(227)),
            boost_width(major_width),
            true,
        ),
        LineworkRole::Lake => (
            Rgba::with_alpha(64, 70, 78, alpha(125)),
            boost_width(minor_width),
            true,
        ),
        LineworkRole::International => (
            Rgba::with_alpha(30, 36, 44, alpha(209)),
            boost_width(minor_width),
            true,
        ),
        LineworkRole::State => (
            Rgba::with_alpha(18, 22, 28, alpha(220)),
            boost_width(minor_width),
            true,
        ),
        LineworkRole::County => (
            Rgba::with_alpha(92, 100, 112, alpha(55)),
            boost_width(minor_width),
            county_visible,
        ),
        LineworkRole::Generic => (fallback, boost_width(fallback_width), true),
    };

    LineworkStyle {
        visible,
        color,
        width,
    }
}

fn clean_atlas_county_linework_visible(mode: ProductVisualMode) -> bool {
    std::env::var("RUSTWX_COUNTY_LINEWORK")
        .ok()
        .map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "0" | "false" | "off" | "no" => false,
            "1" | "true" | "on" | "yes" => true,
            _ => default_clean_atlas_county_linework_visible(mode),
        })
        .unwrap_or_else(|| default_clean_atlas_county_linework_visible(mode))
}

fn default_clean_atlas_county_linework_visible(mode: ProductVisualMode) -> bool {
    matches!(
        mode,
        ProductVisualMode::FilledMeteorology
            | ProductVisualMode::PanelMember
            | ProductVisualMode::ComparisonPanel
    )
}

fn static_linework_width_boost() -> u32 {
    std::env::var("RUSTWX_LINEWORK_WIDTH_BOOST")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(0)
        .clamp(0, 4)
}

fn static_linework_alpha_scale() -> f32 {
    std::env::var("RUSTWX_LINEWORK_ALPHA_SCALE")
        .ok()
        .and_then(|value| value.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(1.0)
        .clamp(0.25, 2.0)
}

fn clean_atlas_chrome(title_anchor: TitleAnchor) -> ChromeStyle {
    ChromeStyle {
        title_anchor,
        title_color: Rgba::new(16, 22, 30),
        subtitle_color: Rgba::new(82, 92, 106),
        frame_color: Some(Rgba::with_alpha(46, 56, 68, 118)),
    }
}

fn clean_atlas_colorbar() -> ColorbarPresentation {
    ColorbarPresentation {
        orientation: ColorbarOrientation::HorizontalBottom,
        frame_color: Rgba::BLACK,
        divider_color: Rgba::with_alpha(0, 0, 0, 185),
        tick_color: Rgba::BLACK,
        label_color: Rgba::BLACK,
    }
}

fn operational_colorbar() -> ColorbarPresentation {
    ColorbarPresentation {
        orientation: ColorbarOrientation::VerticalRight,
        frame_color: Rgba::BLACK,
        divider_color: Rgba::with_alpha(0, 0, 0, 185),
        tick_color: Rgba::BLACK,
        label_color: Rgba::BLACK,
    }
}

fn clean_atlas_layout() -> LayoutMetrics {
    LayoutMetrics {
        margin_x: 18,
        title_h: 44,
        footer_h: 36,
        colorbar_h: 12,
        colorbar_gap: 10,
        colorbar_margin_x: 104,
    }
}

fn clean_atlas_compact_layout() -> LayoutMetrics {
    LayoutMetrics {
        margin_x: 10,
        title_h: 36,
        footer_h: 28,
        colorbar_h: 11,
        colorbar_gap: 9,
        colorbar_margin_x: 50,
    }
}
