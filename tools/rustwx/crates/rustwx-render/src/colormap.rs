use crate::color::{Rgba, lerp_rgba};
use crate::request::DiscreteColorScale;
use serde::{Deserialize, Serialize};

const PALETTE_RESOLUTION_MULTIPLIER: usize = 8;
const LEVEL_RESOLUTION_MULTIPLIER: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LevelDensity {
    pub multiplier: usize,
    pub min_source_level_count: usize,
}

impl Default for LevelDensity {
    fn default() -> Self {
        Self {
            multiplier: 1,
            min_source_level_count: usize::MAX,
        }
    }
}

impl LevelDensity {
    pub const fn fill_default() -> Self {
        Self {
            multiplier: LEVEL_RESOLUTION_MULTIPLIER,
            min_source_level_count: 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderDensity {
    pub fill: LevelDensity,
    pub palette_multiplier: usize,
}

impl Default for RenderDensity {
    fn default() -> Self {
        Self {
            fill: LevelDensity::fill_default(),
            palette_multiplier: PALETTE_RESOLUTION_MULTIPLIER,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LegendMode {
    #[default]
    Stepped,
    SmoothRamp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegendControls {
    pub density: LevelDensity,
    pub mode: LegendMode,
}

impl Default for LegendControls {
    fn default() -> Self {
        Self {
            density: LevelDensity::default(),
            mode: LegendMode::Stepped,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColormapBuildOptions {
    pub render_density: RenderDensity,
    pub legend: LegendControls,
}

impl Default for ColormapBuildOptions {
    fn default() -> Self {
        Self {
            render_density: RenderDensity::default(),
            legend: LegendControls::default(),
        }
    }
}

fn densify_palette_with_multiplier(palette: &[Rgba], multiplier: usize) -> Vec<Rgba> {
    if palette.len() <= 1 || multiplier <= 1 {
        return palette.to_vec();
    }
    let dense_len = (palette.len() - 1) * multiplier + 1;
    lerp_rgba(palette, dense_len)
}

fn densify_levels_with_density(levels: &[f64], density: LevelDensity) -> Vec<f64> {
    if levels.len() < 2 || density.multiplier <= 1 || levels.len() < density.min_source_level_count
    {
        return levels.to_vec();
    }

    let mut dense = Vec::with_capacity((levels.len() - 1) * density.multiplier + 1);
    dense.push(levels[0]);
    for window in levels.windows(2) {
        let lo = window[0];
        let hi = window[1];
        let step = (hi - lo) / density.multiplier as f64;
        if !step.is_finite() || step <= 0.0 {
            continue;
        }
        for i in 1..=density.multiplier {
            dense.push(lo + step * i as f64);
        }
    }
    dense
}

fn sample_listed_palette(palette: &[Rgba], t: f64) -> Rgba {
    if palette.is_empty() {
        return Rgba::TRANSPARENT;
    }
    if palette.len() == 1 {
        return palette[0];
    }
    if !t.is_finite() || t <= 0.0 {
        return palette[0];
    }
    if t >= 1.0 {
        return palette[palette.len() - 1];
    }
    let idx = (t * palette.len() as f64).floor() as usize;
    palette[idx.min(palette.len() - 1)]
}

fn sample_palette_for_levels(palette: &[Rgba], levels: &[f64]) -> Vec<Rgba> {
    if palette.is_empty() || levels.len() < 2 {
        return vec![];
    }

    let min_level = levels[0];
    let max_level = levels[levels.len() - 1];
    let level_span = max_level - min_level;

    levels[..levels.len() - 1]
        .iter()
        .map(|level| {
            let t = if !level_span.is_finite() || level_span <= 0.0 {
                0.0
            } else {
                (*level - min_level) / level_span
            };
            sample_listed_palette(palette, t)
        })
        .collect()
}

/// How to handle values outside the level range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extend {
    Neither,
    Min,
    Max,
    Both,
}

/// A discrete colormap mapping value intervals to colours.
///
/// Given N+1 level boundaries, there are N intervals.
/// `colors` has exactly N entries (one per interval).
/// Optional `under_color` / `over_color` handle values outside the range.
#[derive(Clone, Debug)]
pub struct LeveledColormap {
    pub levels: Vec<f64>,
    pub colors: Vec<Rgba>,
    pub legend_levels: Vec<f64>,
    pub legend_colors: Vec<Rgba>,
    pub under_color: Option<Rgba>,
    pub over_color: Option<Rgba>,
    pub mask_below: Option<f64>,
}

impl LeveledColormap {
    /// Map a data value to a colour.
    pub fn map(&self, value: f64) -> Rgba {
        if value.is_nan() {
            return Rgba::TRANSPARENT;
        }
        if let Some(mb) = self.mask_below {
            if value < mb {
                return Rgba::TRANSPARENT;
            }
        }
        if self.levels.is_empty() || self.colors.is_empty() {
            return Rgba::TRANSPARENT;
        }
        // Below first level
        if value < self.levels[0] {
            return self.under_color.unwrap_or(Rgba::TRANSPARENT);
        }
        let n_intervals = self.levels.len() - 1;
        let idx = self.levels.partition_point(|level| *level <= value);
        if idx <= n_intervals {
            return self.colors[idx.saturating_sub(1).min(self.colors.len() - 1)];
        }
        // Value == last level or above
        if value >= self.levels[n_intervals] {
            return self
                .over_color
                .unwrap_or(self.colors[self.colors.len() - 1]);
        }
        // Exact match on last boundary → last interval
        self.colors[self.colors.len() - 1]
    }

    /// Build from a palette (list of colours) and levels.
    ///
    /// Samples `palette` to produce one colour per interval, matching
    /// matplotlib's behaviour with `contourf(levels=..., cmap=cmap)`.
    pub fn from_palette_with_options(
        palette: &[Rgba],
        levels: &[f64],
        extend: Extend,
        mask_below: Option<f64>,
        options: ColormapBuildOptions,
    ) -> Self {
        let dense_levels = densify_levels_with_density(levels, options.render_density.fill);
        let legend_levels = densify_levels_with_density(levels, options.legend.density);
        let n_intervals = if dense_levels.len() > 1 {
            dense_levels.len() - 1
        } else {
            0
        };
        if n_intervals == 0 || palette.is_empty() {
            return Self {
                levels: dense_levels,
                colors: vec![],
                legend_levels,
                legend_colors: vec![],
                under_color: None,
                over_color: None,
                mask_below,
            };
        }

        // Matplotlib's ListedColormap samples discrete bins by numeric boundary
        // position, not by interval index. That matters for nonlinear level
        // arrays like EHI where low-end bins are intentionally narrower.
        let dense_palette =
            densify_palette_with_multiplier(palette, options.render_density.palette_multiplier);

        let sampled = sample_palette_for_levels(&dense_palette, &dense_levels);
        let legend_colors = sample_palette_for_levels(&dense_palette, &legend_levels);

        let under_color = match extend {
            Extend::Min | Extend::Both => Some(sample_listed_palette(&dense_palette, 0.0)),
            Extend::Neither | Extend::Max => None,
        };
        let over_color = match extend {
            Extend::Max | Extend::Both => Some(sample_listed_palette(&dense_palette, 1.0)),
            Extend::Neither | Extend::Min => None,
        };

        Self {
            levels: dense_levels,
            colors: sampled,
            legend_levels,
            legend_colors,
            under_color,
            over_color,
            mask_below,
        }
    }

    pub fn from_palette(
        palette: &[Rgba],
        levels: &[f64],
        extend: Extend,
        mask_below: Option<f64>,
    ) -> Self {
        Self::from_palette_with_options(
            palette,
            levels,
            extend,
            mask_below,
            ColormapBuildOptions::default(),
        )
    }

    pub fn legend_levels_for_display(&self) -> &[f64] {
        if self.legend_levels.len() > 1 {
            &self.legend_levels
        } else {
            &self.levels
        }
    }

    pub fn legend_colors_for_display(&self) -> &[Rgba] {
        if !self.legend_colors.is_empty() {
            &self.legend_colors
        } else {
            &self.colors
        }
    }
}

pub fn densify_discrete_scale(
    scale: &DiscreteColorScale,
    density: LevelDensity,
) -> DiscreteColorScale {
    let dense_levels = densify_levels_with_density(&scale.levels, density);
    if dense_levels == scale.levels {
        return scale.clone();
    }

    let palette = scale
        .colors
        .iter()
        .copied()
        .map(Into::into)
        .collect::<Vec<Rgba>>();
    let dense_palette = densify_palette_with_multiplier(&palette, PALETTE_RESOLUTION_MULTIPLIER);

    DiscreteColorScale {
        levels: dense_levels,
        colors: dense_palette.into_iter().map(Into::into).collect(),
        extend: scale.extend,
        mask_below: scale.mask_below,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ColormapBuildOptions, LEVEL_RESOLUTION_MULTIPLIER, LevelDensity, LeveledColormap,
        PALETTE_RESOLUTION_MULTIPLIER, densify_discrete_scale, densify_levels_with_density,
        densify_palette_with_multiplier,
    };
    use crate::{Color, DiscreteColorScale, ExtendMode, color::Rgba};

    #[test]
    fn palette_densification_increases_internal_resolution_eightfold() {
        let palette = vec![Rgba::new(0, 0, 0), Rgba::new(255, 255, 255)];
        let dense = densify_palette_with_multiplier(&palette, PALETTE_RESOLUTION_MULTIPLIER);
        assert_eq!(
            dense.len(),
            (palette.len() - 1) * PALETTE_RESOLUTION_MULTIPLIER + 1
        );
        assert_eq!(dense.first().copied(), Some(Rgba::new(0, 0, 0)));
        assert_eq!(dense.last().copied(), Some(Rgba::new(255, 255, 255)));
    }

    #[test]
    fn from_palette_uses_interpolated_midtones_not_just_anchor_colors() {
        let palette = vec![Rgba::new(0, 0, 0), Rgba::new(255, 255, 255)];
        let levels = vec![0.0, 1.0, 2.0];
        let cmap = LeveledColormap::from_palette(&palette, &levels, super::Extend::Neither, None);
        assert_eq!(cmap.colors.len(), 2);
        assert_eq!(cmap.colors[0], Rgba::new(0, 0, 0));
        assert!(cmap.colors[1].r > 0 && cmap.colors[1].r < 255);

        let more_levels = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let smoother = LeveledColormap::from_palette_with_options(
            &palette,
            &more_levels,
            super::Extend::Neither,
            None,
            ColormapBuildOptions {
                render_density: super::RenderDensity {
                    fill: LevelDensity::default(),
                    palette_multiplier: PALETTE_RESOLUTION_MULTIPLIER,
                },
                legend: super::LegendControls::default(),
            },
        );
        assert_eq!(smoother.colors.len(), 4);
        assert!(smoother.colors[1].r > 0 && smoother.colors[1].r < 255);
        assert!(smoother.colors[2].r > 0 && smoother.colors[2].r < 255);
    }

    #[test]
    fn level_densification_increases_visible_fill_intervals() {
        let levels = vec![0.0, 10.0, 20.0];
        assert_eq!(
            densify_levels_with_density(&levels, LevelDensity::fill_default()),
            levels
        );

        let more_levels = vec![0.0, 10.0, 20.0, 30.0, 40.0];
        let dense = densify_levels_with_density(&more_levels, LevelDensity::fill_default());
        assert_eq!(dense.first().copied(), Some(0.0));
        assert_eq!(dense.last().copied(), Some(40.0));
        assert_eq!(
            dense.len(),
            (more_levels.len() - 1) * LEVEL_RESOLUTION_MULTIPLIER + 1
        );
    }

    #[test]
    fn legend_density_defaults_to_original_levels() {
        let palette = vec![Rgba::new(0, 0, 0), Rgba::new(255, 255, 255)];
        let levels = vec![0.0, 10.0, 20.0, 30.0, 40.0];
        let cmap = LeveledColormap::from_palette_with_options(
            &palette,
            &levels,
            super::Extend::Neither,
            None,
            ColormapBuildOptions::default(),
        );
        assert_eq!(cmap.legend_levels, levels);
        assert_eq!(cmap.legend_colors.len(), levels.len() - 1);
        assert!(cmap.levels.len() > cmap.legend_levels.len());
    }

    #[test]
    fn nonlinear_levels_sample_palette_by_numeric_position() {
        let palette = vec![
            Rgba::new(0, 0, 0),
            Rgba::new(10, 10, 10),
            Rgba::new(20, 20, 20),
            Rgba::new(30, 30, 30),
            Rgba::new(40, 40, 40),
            Rgba::new(50, 50, 50),
            Rgba::new(60, 60, 60),
            Rgba::new(70, 70, 70),
        ];
        let levels = vec![0.0, 1.0, 2.0, 4.0, 8.0];

        let cmap = LeveledColormap::from_palette_with_options(
            &palette,
            &levels,
            super::Extend::Both,
            None,
            ColormapBuildOptions {
                render_density: super::RenderDensity {
                    fill: LevelDensity::default(),
                    palette_multiplier: 1,
                },
                legend: super::LegendControls::default(),
            },
        );

        assert_eq!(
            cmap.colors,
            vec![
                Rgba::new(0, 0, 0),
                Rgba::new(10, 10, 10),
                Rgba::new(20, 20, 20),
                Rgba::new(40, 40, 40),
            ]
        );
        assert_eq!(cmap.under_color, Some(Rgba::new(0, 0, 0)));
        assert_eq!(cmap.over_color, Some(Rgba::new(70, 70, 70)));
    }

    #[test]
    fn densify_discrete_scale_inserts_intermediate_intervals() {
        let scale = DiscreteColorScale {
            levels: vec![0.0, 2.0, 4.0],
            colors: vec![Color::rgba(0, 0, 0, 255), Color::rgba(255, 255, 255, 255)],
            extend: ExtendMode::Neither,
            mask_below: None,
        };
        let dense = densify_discrete_scale(
            &scale,
            LevelDensity {
                multiplier: 2,
                min_source_level_count: 2,
            },
        );
        assert_eq!(dense.levels, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert!(dense.colors.len() >= dense.levels.len() - 1);
    }
}
