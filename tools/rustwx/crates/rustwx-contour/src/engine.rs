use std::collections::HashMap;

use crate::field::ScalarField2D;
use crate::geometry::{LineSegment, Point2, Polygon, Ring};
use crate::grid::{CellIndex, Grid2D};
use crate::levels::{ContourLevels, LevelBin, LevelBins, LevelBound};
use crate::topology::{BandPolygon, ContourLayer, ContourSegment, ContourTopology, FillTopology};

const GEOMETRY_EPSILON: f64 = 1.0e-9;
const POINT_KEY_SCALE: f64 = 1.0e9;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SaddleResolver {
    #[default]
    CellAverage,
}

#[derive(Clone, Debug, Default)]
pub struct ContourEngine {
    saddle_resolver: SaddleResolver,
}

impl ContourEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_saddle_resolver(mut self, saddle_resolver: SaddleResolver) -> Self {
        self.saddle_resolver = saddle_resolver;
        self
    }

    pub fn saddle_resolver(&self) -> SaddleResolver {
        self.saddle_resolver
    }

    pub fn extract_isolines<G: Grid2D>(
        &self,
        field: &ScalarField2D<G>,
        levels: &ContourLevels,
    ) -> ContourTopology {
        let mut layers = Vec::with_capacity(levels.values().len());

        for &level in levels.values() {
            let mut segments = Vec::new();

            for cell in field.shape().cells() {
                let Some(sample) = CellSample::from_field(field, cell) else {
                    continue;
                };

                for geometry in self.contour_segments_for_cell(&sample, level) {
                    segments.push(ContourSegment { cell, geometry });
                }
            }

            layers.push(ContourLayer { level, segments });
        }

        ContourTopology { layers }
    }

    pub fn extract_filled_bands<G: Grid2D>(
        &self,
        field: &ScalarField2D<G>,
        bins: &LevelBins,
    ) -> FillTopology {
        let mut polygons = Vec::new();

        for cell in field.shape().cells() {
            let Some(sample) = CellSample::from_field(field, cell) else {
                continue;
            };

            let (min_value, max_value) = sample.value_range_including_center();
            if let Some(bin) = sample.single_bin(bins) {
                if let Some(polygon) = sample.full_cell_polygon() {
                    polygons.push(BandPolygon {
                        bin_index: bin.index,
                        cell,
                        polygon,
                    });
                }
                continue;
            }

            for &bin in bins.bins() {
                if !bin_overlaps_value_range(bin, min_value, max_value) {
                    continue;
                }
                let fragments = self.band_fragments_for_cell(&sample, bin);
                for polygon in merge_fragments(fragments) {
                    polygons.push(BandPolygon {
                        bin_index: bin.index,
                        cell,
                        polygon,
                    });
                }
            }
        }

        FillTopology {
            bins: bins.clone(),
            polygons,
        }
    }

    fn contour_segments_for_cell(&self, sample: &CellSample, level: f64) -> Vec<LineSegment> {
        let mask = sample.mask(level);
        if mask == 0 || mask == 15 {
            return Vec::new();
        }

        let center_is_high = match self.saddle_resolver {
            SaddleResolver::CellAverage => sample.center.value >= level,
        };

        let intersections = sample.edge_intersections(level);
        let mut pairs = [(0usize, 0usize); 2];
        let pair_count = match mask {
            1 | 14 => {
                pairs[0] = (3, 0);
                1
            }
            2 | 13 => {
                pairs[0] = (0, 1);
                1
            }
            3 | 12 => {
                pairs[0] = (3, 1);
                1
            }
            4 | 11 => {
                pairs[0] = (1, 2);
                1
            }
            5 => {
                if center_is_high {
                    pairs[0] = (0, 1);
                    pairs[1] = (2, 3);
                } else {
                    pairs[0] = (3, 0);
                    pairs[1] = (1, 2);
                }
                2
            }
            6 | 9 => {
                pairs[0] = (0, 2);
                1
            }
            7 | 8 => {
                pairs[0] = (3, 2);
                1
            }
            10 => {
                if center_is_high {
                    pairs[0] = (3, 0);
                    pairs[1] = (1, 2);
                } else {
                    pairs[0] = (0, 1);
                    pairs[1] = (2, 3);
                }
                2
            }
            _ => 0,
        };

        let mut segments = Vec::with_capacity(pair_count);
        for &(start_edge, end_edge) in &pairs[..pair_count] {
            let (Some(start), Some(end)) = (intersections[start_edge], intersections[end_edge])
            else {
                continue;
            };

            let segment = LineSegment::new(start, end);
            if segment.length_squared() > GEOMETRY_EPSILON * GEOMETRY_EPSILON {
                segments.push(segment);
            }
        }

        segments
    }

    fn band_fragments_for_cell(&self, sample: &CellSample, bin: LevelBin) -> Vec<Vec<Point2>> {
        let mut fragments = Vec::new();
        for triangle in sample.triangles() {
            if let Some(fragment) = clip_triangle_to_bin(&triangle, bin) {
                fragments.push(fragment);
            }
        }
        fragments
    }
}

#[derive(Clone, Copy, Debug)]
struct CellSample {
    corners: [SamplePoint; 4],
    center: SamplePoint,
}

impl CellSample {
    fn from_field<G: Grid2D>(field: &ScalarField2D<G>, cell: CellIndex) -> Option<Self> {
        let corners = field.grid().cell_corners(cell)?;
        let values = field.cell_values(cell)?;
        if values.iter().any(|value| !value.is_finite()) {
            return None;
        }

        let center = SamplePoint {
            point: field.grid().cell_center(cell)?,
            value: field.cell_center_value(cell)?,
        };

        Some(Self {
            corners: [
                SamplePoint {
                    point: corners[0],
                    value: values[0],
                },
                SamplePoint {
                    point: corners[1],
                    value: values[1],
                },
                SamplePoint {
                    point: corners[2],
                    value: values[2],
                },
                SamplePoint {
                    point: corners[3],
                    value: values[3],
                },
            ],
            center,
        })
    }

    fn mask(&self, level: f64) -> u8 {
        let mut mask = 0u8;
        for (index, corner) in self.corners.iter().enumerate() {
            if corner.value >= level {
                mask |= 1 << index;
            }
        }
        mask
    }

    fn edge_intersections(&self, level: f64) -> [Option<Point2>; 4] {
        [
            edge_intersection(self.corners[0], self.corners[1], level),
            edge_intersection(self.corners[1], self.corners[2], level),
            edge_intersection(self.corners[2], self.corners[3], level),
            edge_intersection(self.corners[3], self.corners[0], level),
        ]
    }

    fn triangles(&self) -> [[SamplePoint; 3]; 4] {
        [
            [self.corners[0], self.corners[1], self.center],
            [self.corners[1], self.corners[2], self.center],
            [self.corners[2], self.corners[3], self.center],
            [self.corners[3], self.corners[0], self.center],
        ]
    }

    fn value_range_including_center(&self) -> (f64, f64) {
        let mut min_value = self.center.value;
        let mut max_value = self.center.value;
        for corner in &self.corners {
            min_value = min_value.min(corner.value);
            max_value = max_value.max(corner.value);
        }
        (min_value, max_value)
    }

    fn all_values_in_bin(&self, bin: LevelBin) -> bool {
        bin.contains(self.center.value)
            && self.corners.iter().all(|corner| bin.contains(corner.value))
    }

    fn single_bin(&self, bins: &LevelBins) -> Option<LevelBin> {
        let bin = *bins.bin(bins.bin_index(self.center.value)?)?;
        self.all_values_in_bin(bin).then_some(bin)
    }

    fn full_cell_polygon(&self) -> Option<Polygon> {
        let ring = sanitize_ring(self.corners.iter().map(|corner| corner.point).collect());
        if ring.len() < 3 {
            return None;
        }
        let polygon = Polygon {
            exterior: Ring { vertices: ring },
            holes: Vec::new(),
        };
        (polygon.area() > GEOMETRY_EPSILON).then_some(polygon)
    }
}

#[derive(Clone, Copy, Debug)]
struct SamplePoint {
    point: Point2,
    value: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PointKey {
    x: i64,
    y: i64,
}

impl PointKey {
    fn new(point: Point2) -> Self {
        Self {
            x: (point.x * POINT_KEY_SCALE).round() as i64,
            y: (point.y * POINT_KEY_SCALE).round() as i64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct EdgeKey {
    a: PointKey,
    b: PointKey,
}

impl EdgeKey {
    fn new(a: PointKey, b: PointKey) -> Self {
        if a.x < b.x || (a.x == b.x && a.y <= b.y) {
            Self { a, b }
        } else {
            Self { a: b, b: a }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BoundaryEdge {
    a: Point2,
    b: Point2,
    a_key: PointKey,
    b_key: PointKey,
}

fn edge_intersection(a: SamplePoint, b: SamplePoint, level: f64) -> Option<Point2> {
    let crosses = (a.value < level && b.value >= level) || (a.value >= level && b.value < level);
    if !crosses {
        return None;
    }

    let delta = b.value - a.value;
    if delta.abs() <= GEOMETRY_EPSILON {
        return None;
    }

    let t = ((level - a.value) / delta).clamp(0.0, 1.0);
    Some(a.point.lerp(b.point, t))
}

fn clip_triangle_to_bin(triangle: &[SamplePoint; 3], bin: LevelBin) -> Option<Vec<Point2>> {
    let mut polygon = triangle.to_vec();

    if let LevelBound::Finite(lower) = bin.lower {
        polygon = clip_polygon(polygon, lower, ClipRule::LowerInclusive);
    }
    if polygon.len() < 3 {
        return None;
    }

    if let LevelBound::Finite(upper) = bin.upper {
        let upper_rule = if bin.upper_inclusive {
            ClipRule::UpperInclusive
        } else {
            ClipRule::UpperExclusive
        };
        polygon = clip_polygon(polygon, upper, upper_rule);
    }
    if polygon.len() < 3 {
        return None;
    }

    let vertices = sanitize_ring(
        polygon
            .into_iter()
            .map(|vertex| vertex.point)
            .collect::<Vec<Point2>>(),
    );
    if vertices.len() < 3 {
        return None;
    }

    let ring = Ring {
        vertices: vertices.clone(),
    };
    if ring.area() <= GEOMETRY_EPSILON {
        return None;
    }

    Some(vertices)
}

#[derive(Clone, Copy, Debug)]
enum ClipRule {
    LowerInclusive,
    UpperExclusive,
    UpperInclusive,
}

fn clip_polygon(vertices: Vec<SamplePoint>, boundary: f64, rule: ClipRule) -> Vec<SamplePoint> {
    if vertices.is_empty() {
        return Vec::new();
    }

    let mut output = Vec::new();
    let mut previous = *vertices.last().expect("non-empty vertices");
    let mut previous_inside = is_inside(previous.value, boundary, rule);

    for current in vertices {
        let current_inside = is_inside(current.value, boundary, rule);

        if current_inside {
            if !previous_inside {
                output.push(interpolate(previous, current, boundary));
            }
            output.push(current);
        } else if previous_inside {
            output.push(interpolate(previous, current, boundary));
        }

        previous = current;
        previous_inside = current_inside;
    }

    output
}

fn is_inside(value: f64, boundary: f64, rule: ClipRule) -> bool {
    match rule {
        ClipRule::LowerInclusive => value >= boundary,
        ClipRule::UpperExclusive => value < boundary,
        ClipRule::UpperInclusive => value <= boundary,
    }
}

fn bin_overlaps_value_range(bin: LevelBin, min_value: f64, max_value: f64) -> bool {
    let lower_ok = match bin.lower {
        LevelBound::NegInfinity => true,
        LevelBound::Finite(boundary) => max_value >= boundary,
        LevelBound::PosInfinity => false,
    };
    let upper_ok = match bin.upper {
        LevelBound::NegInfinity => false,
        LevelBound::Finite(boundary) => {
            if bin.upper_inclusive {
                min_value <= boundary
            } else {
                min_value < boundary
            }
        }
        LevelBound::PosInfinity => true,
    };
    lower_ok && upper_ok
}

fn interpolate(a: SamplePoint, b: SamplePoint, boundary: f64) -> SamplePoint {
    let delta = b.value - a.value;
    if delta.abs() <= GEOMETRY_EPSILON {
        return SamplePoint {
            point: a.point,
            value: boundary,
        };
    }

    let t = ((boundary - a.value) / delta).clamp(0.0, 1.0);
    SamplePoint {
        point: a.point.lerp(b.point, t),
        value: boundary,
    }
}

fn sanitize_ring(vertices: Vec<Point2>) -> Vec<Point2> {
    let mut ring = Vec::with_capacity(vertices.len());
    for vertex in vertices {
        if ring
            .last()
            .is_none_or(|previous| !same_point(*previous, vertex))
        {
            ring.push(vertex);
        }
    }

    if ring.len() >= 2 && same_point(ring[0], *ring.last().expect("ring has len >= 2")) {
        ring.pop();
    }

    ring
}

fn merge_fragments(fragments: Vec<Vec<Point2>>) -> Vec<Polygon> {
    if fragments.is_empty() {
        return Vec::new();
    }

    let mut edge_counts: HashMap<EdgeKey, usize> = HashMap::new();
    let mut edges = Vec::new();

    for fragment in fragments {
        let ring = sanitize_ring(fragment);
        if ring.len() < 3 {
            continue;
        }

        for index in 0..ring.len() {
            let start = ring[index];
            let end = ring[(index + 1) % ring.len()];
            if same_point(start, end) {
                continue;
            }

            let start_key = PointKey::new(start);
            let end_key = PointKey::new(end);
            *edge_counts
                .entry(EdgeKey::new(start_key, end_key))
                .or_insert(0) += 1;
            edges.push(BoundaryEdge {
                a: start,
                b: end,
                a_key: start_key,
                b_key: end_key,
            });
        }
    }

    let boundary_edges: Vec<BoundaryEdge> = edges
        .into_iter()
        .filter(|edge| edge_counts[&EdgeKey::new(edge.a_key, edge.b_key)] == 1)
        .collect();

    let mut adjacency: HashMap<PointKey, Vec<usize>> = HashMap::new();
    for (index, edge) in boundary_edges.iter().enumerate() {
        adjacency.entry(edge.a_key).or_default().push(index);
        adjacency.entry(edge.b_key).or_default().push(index);
    }

    let mut used = vec![false; boundary_edges.len()];
    let mut polygons = Vec::new();

    for start_edge in 0..boundary_edges.len() {
        if used[start_edge] {
            continue;
        }

        let edge = boundary_edges[start_edge];
        let mut ring = vec![edge.a];
        let start_key = edge.a_key;
        let mut current_key = edge.b_key;
        let mut current_point = edge.b;
        used[start_edge] = true;
        ring.push(current_point);

        while current_key != start_key {
            let Some(next_edge_index) = adjacency
                .get(&current_key)
                .and_then(|indices| indices.iter().copied().find(|index| !used[*index]))
            else {
                ring.clear();
                break;
            };

            used[next_edge_index] = true;
            let next_edge = boundary_edges[next_edge_index];
            let (next_key, next_point) = if next_edge.a_key == current_key {
                (next_edge.b_key, next_edge.b)
            } else {
                (next_edge.a_key, next_edge.a)
            };

            if !same_point(current_point, next_point) {
                ring.push(next_point);
            }
            current_key = next_key;
            current_point = next_point;
        }

        let ring = sanitize_ring(ring);
        if ring.len() < 3 {
            continue;
        }

        let polygon = Polygon {
            exterior: Ring { vertices: ring },
            holes: Vec::new(),
        };
        if polygon.area() > GEOMETRY_EPSILON {
            polygons.push(polygon);
        }
    }

    polygons
}

fn same_point(a: Point2, b: Point2) -> bool {
    (a.x - b.x).abs() <= GEOMETRY_EPSILON && (a.y - b.y).abs() <= GEOMETRY_EPSILON
}

#[cfg(test)]
mod tests {
    use super::{ContourEngine, SaddleResolver};
    use crate::{
        ContourLevels, ExtendMode, GridShape, LevelBins, Point2, ProjectedGrid, RectilinearGrid,
        ScalarField2D,
    };

    fn single_cell_field(corners: [f64; 4]) -> ScalarField2D<RectilinearGrid> {
        let grid = RectilinearGrid::new(vec![0.0, 1.0], vec![0.0, 1.0]).expect("grid");
        let [south_west, south_east, north_east, north_west] = corners;
        ScalarField2D::new(grid, vec![south_west, south_east, north_west, north_east])
            .expect("field")
    }

    fn sort_segment_points(topology: &crate::ContourTopology) -> Vec<Vec<(f64, f64)>> {
        let mut segments = topology.layers[0]
            .segments
            .iter()
            .map(|segment| {
                let mut endpoints = vec![
                    (
                        round3(segment.geometry.start.x),
                        round3(segment.geometry.start.y),
                    ),
                    (
                        round3(segment.geometry.end.x),
                        round3(segment.geometry.end.y),
                    ),
                ];
                endpoints.sort_by(|left, right| left.partial_cmp(right).expect("finite points"));
                endpoints
            })
            .collect::<Vec<_>>();
        segments.sort_by(|left, right| left.partial_cmp(right).expect("finite points"));
        segments
    }

    fn round3(value: f64) -> f64 {
        (value * 1000.0).round() / 1000.0
    }

    #[test]
    fn rejects_non_monotonic_rectilinear_axis() {
        let error = RectilinearGrid::new(vec![0.0, 1.0, 1.0], vec![0.0, 1.0]).unwrap_err();
        assert!(matches!(
            error,
            crate::ContourError::AxisNotStrictlyMonotonic { axis: "x", .. }
        ));
    }

    #[test]
    fn resolves_ambiguous_case_5_when_center_is_high() {
        let field = single_cell_field([2.0, 0.0, 2.0, 0.5]);
        let levels = ContourLevels::new(vec![1.0]).expect("levels");
        let topology = ContourEngine::new()
            .with_saddle_resolver(SaddleResolver::CellAverage)
            .extract_isolines(&field, &levels);

        assert_eq!(
            sort_segment_points(&topology),
            vec![
                vec![(0.0, 0.667), (0.333, 1.0)],
                vec![(0.5, 0.0), (1.0, 0.5)],
            ]
        );
    }

    #[test]
    fn resolves_ambiguous_case_5_when_center_is_low() {
        let field = single_cell_field([2.0, 0.0, 1.2, 0.0]);
        let levels = ContourLevels::new(vec![1.0]).expect("levels");
        let topology = ContourEngine::new().extract_isolines(&field, &levels);

        assert_eq!(
            sort_segment_points(&topology),
            vec![
                vec![(0.0, 0.5), (0.5, 0.0)],
                vec![(0.833, 1.0), (1.0, 0.833)],
            ]
        );
    }

    #[test]
    fn uniform_threshold_cell_assigns_to_upper_band_only() {
        let field = single_cell_field([1.0, 1.0, 1.0, 1.0]);
        let bins = LevelBins::bounded(vec![0.0, 1.0, 2.0]).expect("bins");
        let topology = ContourEngine::new().extract_filled_bands(&field, &bins);

        assert_eq!(topology.polygons.len(), 1);
        assert_eq!(topology.polygons[0].bin_index, 1);
        assert_eq!(round3(topology.polygons[0].polygon.area()), 1.0);
    }

    #[test]
    fn extended_bins_use_consistent_boundary_assignment() {
        let bins = LevelBins::with_extend(vec![0.0, 10.0, 20.0], ExtendMode::Both).expect("bins");
        assert_eq!(bins.bin_index(-5.0), Some(0));
        assert_eq!(bins.bin_index(0.0), Some(1));
        assert_eq!(bins.bin_index(10.0), Some(2));
        assert_eq!(bins.bin_index(20.0), Some(3));
    }

    #[test]
    fn projected_grid_returns_projected_geometry() {
        let shape = GridShape::new(2, 2).expect("shape");
        let grid = ProjectedGrid::new(
            shape,
            vec![
                Point2::new(10.0, 100.0),
                Point2::new(20.0, 100.0),
                Point2::new(10.0, 110.0),
                Point2::new(20.0, 110.0),
            ],
        )
        .expect("grid");
        let field = ScalarField2D::new(grid, vec![0.0, 0.0, 2.0, 2.0]).expect("field");
        let topology = ContourEngine::new()
            .extract_isolines(&field, &ContourLevels::new(vec![1.0]).expect("levels"));

        let segments = sort_segment_points(&topology);
        assert_eq!(segments, vec![vec![(10.0, 105.0), (20.0, 105.0)]]);
    }
}
