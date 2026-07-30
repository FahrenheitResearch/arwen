use crate::RustwxRenderError;
use crate::color::Rgba;
use crate::colormap::LeveledColormap;
use crate::presentation::{LineworkRole, PolygonRole};
use crate::request::{
    Color, ColorScale, DiscreteColorScale, ExtendMode, Field2D, ProjectedDomain,
    ProjectedLineOverlay, ProjectedPolygonFill,
};
use rustwx_contour::{
    ContourEngine, ContourLevels, ExtendMode as ContourExtendMode, GridShape as ContourGridShape,
    LevelBin, LevelBins, LevelBound, Point2, Polygon, ProjectedGrid as ContourProjectedGrid,
    ScalarField2D,
};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectedContourLineStyle {
    pub color: Color,
    pub width: u32,
}

impl Default for ProjectedContourLineStyle {
    fn default() -> Self {
        Self {
            color: Color::BLACK,
            width: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ProjectedContourGeometry {
    pub fills: Vec<ProjectedPolygonFill>,
    pub lines: Vec<ProjectedLineOverlay>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ProjectedContourGeometryTiming {
    pub projected_points_ms: u128,
    pub scalar_field_ms: u128,
    pub fill_topology_ms: u128,
    pub fill_geometry_ms: u128,
    pub line_topology_ms: u128,
    pub line_geometry_ms: u128,
    pub total_ms: u128,
}

pub fn build_projected_contour_geometry(
    field: &Field2D,
    projected_domain: &ProjectedDomain,
    scale: &ColorScale,
    line_levels: &[f64],
    line_style: ProjectedContourLineStyle,
) -> Result<ProjectedContourGeometry, RustwxRenderError> {
    build_projected_contour_geometry_profile(
        field,
        projected_domain,
        scale,
        line_levels,
        line_style,
    )
    .map(|(geometry, _)| geometry)
}

pub fn build_projected_contour_geometry_profile(
    field: &Field2D,
    projected_domain: &ProjectedDomain,
    scale: &ColorScale,
    line_levels: &[f64],
    line_style: ProjectedContourLineStyle,
) -> Result<(ProjectedContourGeometry, ProjectedContourGeometryTiming), RustwxRenderError> {
    let total_start = Instant::now();
    let expected = field.grid.shape.len();
    if projected_domain.x.len() != expected {
        return Err(RustwxRenderError::LayerShapeMismatch {
            layer: "projected_contour_fill_x",
            expected,
            actual: projected_domain.x.len(),
        });
    }
    if projected_domain.y.len() != expected {
        return Err(RustwxRenderError::LayerShapeMismatch {
            layer: "projected_contour_fill_y",
            expected,
            actual: projected_domain.y.len(),
        });
    }

    let projected_points_start = Instant::now();
    let shape = ContourGridShape::new(field.grid.shape.nx, field.grid.shape.ny)
        .map_err(|err| RustwxRenderError::ContourTopology(err.to_string()))?;
    let points = projected_domain
        .x
        .iter()
        .zip(&projected_domain.y)
        .map(|(&x, &y)| Point2::new(x, y))
        .collect::<Vec<_>>();
    let grid = ContourProjectedGrid::new(shape, points)
        .map_err(|err| RustwxRenderError::ContourTopology(err.to_string()))?;
    let projected_points_ms = projected_points_start.elapsed().as_millis();
    let scalar_field_start = Instant::now();
    let values = field
        .values
        .iter()
        .map(|&value| value as f64)
        .collect::<Vec<_>>();
    let scalar = ScalarField2D::new(grid, values)
        .map_err(|err| RustwxRenderError::ContourTopology(err.to_string()))?;
    let scalar_field_ms = scalar_field_start.elapsed().as_millis();
    let discrete = scale.resolved_discrete();
    let leveled = leveled_colormap(&discrete);
    let bins = LevelBins::with_extend(
        discrete.levels.clone(),
        contour_extend_mode(discrete.extend),
    )
    .map_err(|err| RustwxRenderError::ContourTopology(err.to_string()))?;

    let engine = ContourEngine::new();
    let fill_topology_start = Instant::now();
    let fill_topology = engine.extract_filled_bands(&scalar, &bins);
    let fill_topology_ms = fill_topology_start.elapsed().as_millis();
    let fill_geometry_start = Instant::now();
    let fills = fill_topology
        .polygons
        .iter()
        .filter_map(|band| {
            let bin = fill_topology.bin(band.bin_index)?;
            if bin_is_masked(bin, discrete.mask_below) {
                return None;
            }
            let color = color_for_bin(&leveled, bin);
            let rings = polygon_to_rings(&band.polygon);
            if rings.is_empty() {
                return None;
            }
            Some(ProjectedPolygonFill {
                rings,
                color,
                role: PolygonRole::Generic,
            })
        })
        .collect();
    let fill_geometry_ms = fill_geometry_start.elapsed().as_millis();

    let mut lines = Vec::new();
    let mut line_topology_ms = 0;
    let mut line_geometry_ms = 0;
    if !line_levels.is_empty() {
        let line_topology_start = Instant::now();
        let levels = ContourLevels::new(line_levels.to_vec())
            .map_err(|err| RustwxRenderError::ContourTopology(err.to_string()))?;
        let topology = engine.extract_isolines(&scalar, &levels);
        line_topology_ms = line_topology_start.elapsed().as_millis();
        let line_geometry_start = Instant::now();
        for layer in topology.layers {
            let _ = layer.level;
            for segment in layer.segments {
                lines.push(ProjectedLineOverlay {
                    points: vec![
                        (segment.geometry.start.x, segment.geometry.start.y),
                        (segment.geometry.end.x, segment.geometry.end.y),
                    ],
                    color: line_style.color,
                    width: line_style.width.max(1),
                    role: LineworkRole::Generic,
                });
            }
        }
        line_geometry_ms = line_geometry_start.elapsed().as_millis();
    }

    Ok((
        ProjectedContourGeometry { fills, lines },
        ProjectedContourGeometryTiming {
            projected_points_ms,
            scalar_field_ms,
            fill_topology_ms,
            fill_geometry_ms,
            line_topology_ms,
            line_geometry_ms,
            total_ms: total_start.elapsed().as_millis(),
        },
    ))
}

fn contour_extend_mode(mode: ExtendMode) -> ContourExtendMode {
    match mode {
        ExtendMode::Neither => ContourExtendMode::None,
        ExtendMode::Min => ContourExtendMode::Below,
        ExtendMode::Max => ContourExtendMode::Above,
        ExtendMode::Both => ContourExtendMode::Both,
    }
}

fn leveled_colormap(scale: &DiscreteColorScale) -> LeveledColormap {
    let palette = scale
        .colors
        .iter()
        .copied()
        .map(Rgba::from)
        .collect::<Vec<_>>();
    LeveledColormap::from_palette(
        &palette,
        &scale.levels,
        scale.extend.into(),
        scale.mask_below,
    )
}

fn color_for_bin(scale: &LeveledColormap, bin: &LevelBin) -> Color {
    match (bin.lower, bin.upper) {
        (LevelBound::NegInfinity, _) => scale
            .under_color
            .unwrap_or_else(|| scale.colors.first().copied().unwrap_or(Rgba::TRANSPARENT))
            .into(),
        (_, LevelBound::PosInfinity) => scale
            .over_color
            .unwrap_or_else(|| scale.colors.last().copied().unwrap_or(Rgba::TRANSPARENT))
            .into(),
        (LevelBound::Finite(lower), LevelBound::Finite(_)) => {
            let sample_value = match bin.upper {
                LevelBound::Finite(upper) if upper > lower => (lower + upper) * 0.5,
                _ => lower,
            };
            scale.map(sample_value).into()
        }
        _ => Rgba::TRANSPARENT.into(),
    }
}

fn bin_is_masked(bin: &LevelBin, mask_below: Option<f64>) -> bool {
    let Some(mask_below) = mask_below else {
        return false;
    };
    match bin.upper {
        LevelBound::NegInfinity => true,
        LevelBound::Finite(upper) => upper <= mask_below,
        LevelBound::PosInfinity => false,
    }
}

fn polygon_to_rings(polygon: &Polygon) -> Vec<Vec<(f64, f64)>> {
    let mut rings = Vec::with_capacity(1 + polygon.holes.len());
    let exterior = ring_to_points(&polygon.exterior.vertices);
    if exterior.len() >= 3 {
        rings.push(exterior);
    }
    for hole in &polygon.holes {
        let ring = ring_to_points(&hole.vertices);
        if ring.len() >= 3 {
            rings.push(ring);
        }
    }
    rings
}

fn ring_to_points(vertices: &[Point2]) -> Vec<(f64, f64)> {
    vertices
        .iter()
        .filter(|point| point.x.is_finite() && point.y.is_finite())
        .map(|point| (point.x, point.y))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Field2D, GridShape, LatLonGrid, ProductKey, ProjectedExtent};

    fn sample_field() -> Field2D {
        let grid = LatLonGrid::new(
            GridShape::new(3, 3).unwrap(),
            vec![35.0, 35.0, 35.0, 36.0, 36.0, 36.0, 37.0, 37.0, 37.0],
            vec![
                -99.0, -98.0, -97.0, -99.0, -98.0, -97.0, -99.0, -98.0, -97.0,
            ],
        )
        .unwrap();
        Field2D::new(
            ProductKey::named("stp_fixed"),
            "dimensionless",
            grid,
            vec![0.0, 0.5, 1.5, 0.5, 2.0, 3.5, 1.0, 3.0, 6.0],
        )
        .unwrap()
    }

    fn sample_domain() -> ProjectedDomain {
        ProjectedDomain {
            x: vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
            y: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            extent: ProjectedExtent {
                x_min: -1.0,
                x_max: 1.0,
                y_min: 0.0,
                y_max: 2.0,
            },
        }
    }

    #[test]
    fn projected_contour_geometry_builds_fill_polygons_and_lines() {
        let geometry = build_projected_contour_geometry(
            &sample_field(),
            &sample_domain(),
            &ColorScale::Weather(crate::weather::WeatherPreset::Stp),
            &[1.0, 3.0, 5.0],
            ProjectedContourLineStyle {
                color: Color::BLACK,
                width: 2,
            },
        )
        .unwrap();

        assert!(!geometry.fills.is_empty());
        assert!(!geometry.lines.is_empty());
    }
}
