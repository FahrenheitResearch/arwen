use crate::error::CrossSectionError;

const EARTH_RADIUS_KM: f64 = 6_371.0;

/// A latitude/longitude point in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoPoint {
    pub lat_deg: f64,
    pub lon_deg: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoBounds {
    pub west_lon_deg: f64,
    pub east_lon_deg: f64,
    pub south_lat_deg: f64,
    pub north_lat_deg: f64,
}

impl GeoBounds {
    pub fn new(
        west_lon_deg: f64,
        east_lon_deg: f64,
        south_lat_deg: f64,
        north_lat_deg: f64,
    ) -> Result<Self, CrossSectionError> {
        if !west_lon_deg.is_finite()
            || !east_lon_deg.is_finite()
            || !south_lat_deg.is_finite()
            || !north_lat_deg.is_finite()
            || south_lat_deg < -90.0
            || north_lat_deg > 90.0
            || south_lat_deg >= north_lat_deg
            || west_lon_deg >= east_lon_deg
        {
            return Err(CrossSectionError::InvalidCoordinate);
        }
        Ok(Self {
            west_lon_deg,
            east_lon_deg,
            south_lat_deg,
            north_lat_deg,
        })
    }

    pub fn center(self) -> Result<GeoPoint, CrossSectionError> {
        GeoPoint::new(
            (self.south_lat_deg + self.north_lat_deg) * 0.5,
            (self.west_lon_deg + self.east_lon_deg) * 0.5,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepresentativeRouteStrategy {
    LongestAxisMidline,
    FarthestPair,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RepresentativeRoute {
    pub start: GeoPoint,
    pub end: GeoPoint,
}

impl GeoPoint {
    /// Creates a validated point and normalizes longitude into [-180, 180).
    pub fn new(lat_deg: f64, lon_deg: f64) -> Result<Self, CrossSectionError> {
        if !lat_deg.is_finite() || !lon_deg.is_finite() || !(-90.0..=90.0).contains(&lat_deg) {
            return Err(CrossSectionError::InvalidCoordinate);
        }

        Ok(Self {
            lat_deg,
            lon_deg: normalize_longitude(lon_deg),
        })
    }

    fn from_radians(lat_rad: f64, lon_rad: f64) -> Self {
        Self {
            lat_deg: lat_rad.to_degrees(),
            lon_deg: normalize_longitude(lon_rad.to_degrees()),
        }
    }

    fn lat_rad(self) -> f64 {
        self.lat_deg.to_radians()
    }

    fn lon_rad(self) -> f64 {
        self.lon_deg.to_radians()
    }
}

pub fn representative_route_for_bounds(
    bounds: GeoBounds,
    strategy: RepresentativeRouteStrategy,
) -> Result<RepresentativeRoute, CrossSectionError> {
    match strategy {
        RepresentativeRouteStrategy::LongestAxisMidline => {
            let center = bounds.center()?;
            let west = GeoPoint::new(center.lat_deg, bounds.west_lon_deg)?;
            let east = GeoPoint::new(center.lat_deg, bounds.east_lon_deg)?;
            let south = GeoPoint::new(bounds.south_lat_deg, center.lon_deg)?;
            let north = GeoPoint::new(bounds.north_lat_deg, center.lon_deg)?;

            let width_km = haversine_distance_km(west, east);
            let height_km = haversine_distance_km(south, north);
            if width_km >= height_km {
                Ok(RepresentativeRoute {
                    start: west,
                    end: east,
                })
            } else {
                Ok(RepresentativeRoute {
                    start: south,
                    end: north,
                })
            }
        }
        RepresentativeRouteStrategy::FarthestPair => representative_route_for_cluster(
            &[
                GeoPoint::new(bounds.south_lat_deg, bounds.west_lon_deg)?,
                GeoPoint::new(bounds.south_lat_deg, bounds.east_lon_deg)?,
                GeoPoint::new(bounds.north_lat_deg, bounds.west_lon_deg)?,
                GeoPoint::new(bounds.north_lat_deg, bounds.east_lon_deg)?,
            ],
            RepresentativeRouteStrategy::FarthestPair,
        ),
    }
}

pub fn representative_route_for_cluster(
    points: &[GeoPoint],
    strategy: RepresentativeRouteStrategy,
) -> Result<RepresentativeRoute, CrossSectionError> {
    if points.len() < 2 {
        return Err(CrossSectionError::TooFewWaypoints);
    }

    match strategy {
        RepresentativeRouteStrategy::LongestAxisMidline => {
            let mut west = f64::INFINITY;
            let mut east = f64::NEG_INFINITY;
            let mut south = f64::INFINITY;
            let mut north = f64::NEG_INFINITY;
            for point in points {
                west = west.min(point.lon_deg);
                east = east.max(point.lon_deg);
                south = south.min(point.lat_deg);
                north = north.max(point.lat_deg);
            }
            representative_route_for_bounds(
                GeoBounds::new(west, east, south, north)?,
                RepresentativeRouteStrategy::LongestAxisMidline,
            )
        }
        RepresentativeRouteStrategy::FarthestPair => {
            let mut best: Option<(usize, usize, f64)> = None;
            for i in 0..points.len() {
                for j in i + 1..points.len() {
                    let distance = haversine_distance_km(points[i], points[j]);
                    if best
                        .map(|(_, _, best_distance)| distance > best_distance)
                        .unwrap_or(true)
                    {
                        best = Some((i, j, distance));
                    }
                }
            }
            let Some((start_index, end_index, _)) = best else {
                return Err(CrossSectionError::DegeneratePath);
            };
            Ok(RepresentativeRoute {
                start: points[start_index],
                end: points[end_index],
            })
        }
    }
}

/// A polyline path used to define the horizontal cross-section trace.
#[derive(Debug, Clone, PartialEq)]
pub struct SectionPath {
    waypoints: Vec<GeoPoint>,
}

impl SectionPath {
    pub fn new(waypoints: Vec<GeoPoint>) -> Result<Self, CrossSectionError> {
        if waypoints.len() < 2 {
            return Err(CrossSectionError::TooFewWaypoints);
        }

        let path = Self { waypoints };
        if path.total_distance_km() <= f64::EPSILON {
            return Err(CrossSectionError::DegeneratePath);
        }
        Ok(path)
    }

    pub fn endpoints(start: GeoPoint, end: GeoPoint) -> Result<Self, CrossSectionError> {
        Self::new(vec![start, end])
    }

    pub fn waypoints(&self) -> &[GeoPoint] {
        &self.waypoints
    }

    pub fn total_distance_km(&self) -> f64 {
        self.segment_lengths_km().iter().sum()
    }

    pub fn sample_count(&self, count: usize) -> Result<SampledPath, CrossSectionError> {
        if count < 2 {
            return Err(CrossSectionError::InvalidSampleCount);
        }

        let total_distance = self.total_distance_km();
        if total_distance <= f64::EPSILON {
            return Err(CrossSectionError::DegeneratePath);
        }

        let step = total_distance / (count as f64 - 1.0);
        let distances = (0..count).map(|idx| idx as f64 * step).collect::<Vec<_>>();
        self.sample_distances(&distances)
    }

    pub fn sample_spacing_km(&self, spacing_km: f64) -> Result<SampledPath, CrossSectionError> {
        if !spacing_km.is_finite() || spacing_km <= 0.0 {
            return Err(CrossSectionError::InvalidSpacing);
        }

        let total_distance = self.total_distance_km();
        if total_distance <= f64::EPSILON {
            return Err(CrossSectionError::DegeneratePath);
        }

        let count = ((total_distance / spacing_km).ceil() as usize)
            .saturating_add(1)
            .max(2);
        self.sample_count(count)
    }

    pub fn sample_distances(&self, distances_km: &[f64]) -> Result<SampledPath, CrossSectionError> {
        if distances_km.len() < 2 {
            return Err(CrossSectionError::InvalidSampleCount);
        }
        if distances_km.iter().any(|distance| !distance.is_finite()) {
            return Err(CrossSectionError::NonMonotonicDistances);
        }

        let total_distance = self.total_distance_km();
        let segment_lengths = self.segment_lengths_km();
        let cumulative = cumulative_lengths(&segment_lengths);

        let mut samples = Vec::with_capacity(distances_km.len());
        for &target_distance in distances_km {
            let clamped = target_distance.clamp(0.0, total_distance);
            let segment_index = locate_segment(&cumulative, clamped);
            let start = self.waypoints[segment_index];
            let end = self.waypoints[segment_index + 1];
            let seg_start = cumulative[segment_index];
            let seg_len = segment_lengths[segment_index];
            let seg_fraction = if seg_len <= f64::EPSILON {
                0.0
            } else {
                (clamped - seg_start) / seg_len
            };

            samples.push(PathSample {
                point: intermediate_point(start, end, seg_fraction),
                distance_km: clamped,
                bearing_deg: initial_bearing_deg(start, end),
                segment_index,
                segment_fraction: seg_fraction,
            });
        }

        Ok(SampledPath {
            samples,
            total_distance_km: total_distance,
        })
    }

    fn segment_lengths_km(&self) -> Vec<f64> {
        self.waypoints
            .windows(2)
            .map(|pair| haversine_distance_km(pair[0], pair[1]))
            .collect()
    }
}

/// A single sampled location along a cross-section path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathSample {
    pub point: GeoPoint,
    pub distance_km: f64,
    pub bearing_deg: f64,
    pub segment_index: usize,
    pub segment_fraction: f64,
}

/// Evenly or custom-sampled points along a [`SectionPath`].
#[derive(Debug, Clone, PartialEq)]
pub struct SampledPath {
    pub samples: Vec<PathSample>,
    pub total_distance_km: f64,
}

impl SampledPath {
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn distances_km(&self) -> Vec<f64> {
        self.samples
            .iter()
            .map(|sample| sample.distance_km)
            .collect()
    }

    pub fn bearings_deg(&self) -> Vec<f64> {
        self.samples
            .iter()
            .map(|sample| sample.bearing_deg)
            .collect()
    }

    pub fn points(&self) -> Vec<GeoPoint> {
        self.samples.iter().map(|sample| sample.point).collect()
    }
}

/// Great-circle distance between two points in kilometers.
pub fn haversine_distance_km(a: GeoPoint, b: GeoPoint) -> f64 {
    let lat1 = a.lat_rad();
    let lat2 = b.lat_rad();
    let dlat = lat2 - lat1;
    let dlon = b.lon_rad() - a.lon_rad();

    let hav = (dlat * 0.5).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon * 0.5).sin().powi(2);
    let central_angle = 2.0 * hav.sqrt().asin();
    EARTH_RADIUS_KM * central_angle
}

/// Initial bearing from the first point toward the second, in degrees clockwise from north.
pub fn initial_bearing_deg(a: GeoPoint, b: GeoPoint) -> f64 {
    let lat1 = a.lat_rad();
    let lat2 = b.lat_rad();
    let dlon = b.lon_rad() - a.lon_rad();

    let y = dlon.sin() * lat2.cos();
    let x = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * dlon.cos();
    y.atan2(x).to_degrees().rem_euclid(360.0)
}

/// Great-circle interpolation between two points.
pub fn intermediate_point(a: GeoPoint, b: GeoPoint, fraction: f64) -> GeoPoint {
    let fraction = fraction.clamp(0.0, 1.0);
    if fraction <= f64::EPSILON {
        return a;
    }
    if (1.0 - fraction) <= f64::EPSILON {
        return b;
    }

    let lat1 = a.lat_rad();
    let lon1 = a.lon_rad();
    let lat2 = b.lat_rad();
    let lon2 = b.lon_rad();

    let delta = haversine_distance_km(a, b) / EARTH_RADIUS_KM;
    if delta <= f64::EPSILON {
        return a;
    }

    let sin_delta = delta.sin();
    let a_weight = ((1.0 - fraction) * delta).sin() / sin_delta;
    let b_weight = (fraction * delta).sin() / sin_delta;

    let x = a_weight * lat1.cos() * lon1.cos() + b_weight * lat2.cos() * lon2.cos();
    let y = a_weight * lat1.cos() * lon1.sin() + b_weight * lat2.cos() * lon2.sin();
    let z = a_weight * lat1.sin() + b_weight * lat2.sin();

    let lat = z.atan2((x * x + y * y).sqrt());
    let lon = y.atan2(x);
    GeoPoint::from_radians(lat, lon)
}

fn cumulative_lengths(segment_lengths: &[f64]) -> Vec<f64> {
    let mut cumulative = Vec::with_capacity(segment_lengths.len() + 1);
    cumulative.push(0.0);
    for &segment_length in segment_lengths {
        let last = *cumulative.last().unwrap_or(&0.0);
        cumulative.push(last + segment_length);
    }
    cumulative
}

fn locate_segment(cumulative_lengths: &[f64], distance_km: f64) -> usize {
    let last_segment = cumulative_lengths.len().saturating_sub(2);
    for segment_index in 0..=last_segment {
        if distance_km <= cumulative_lengths[segment_index + 1] || segment_index == last_segment {
            return segment_index;
        }
    }
    last_segment
}

fn normalize_longitude(lon_deg: f64) -> f64 {
    ((lon_deg + 180.0).rem_euclid(360.0)) - 180.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampled_path_preserves_endpoints_and_monotonic_distance() {
        let path = SectionPath::endpoints(
            GeoPoint::new(39.7392, -104.9903).unwrap(),
            GeoPoint::new(41.8781, -87.6298).unwrap(),
        )
        .unwrap();

        let sampled = path.sample_count(5).unwrap();
        assert_eq!(sampled.len(), 5);
        assert_eq!(sampled.samples.first().unwrap().point, path.waypoints()[0]);
        assert_eq!(sampled.samples.last().unwrap().point, path.waypoints()[1]);

        let distances = sampled.distances_km();
        assert!(distances.windows(2).all(|pair| pair[1] > pair[0]));
        assert!((distances.last().copied().unwrap() - sampled.total_distance_km).abs() < 1e-6);
    }

    #[test]
    fn spacing_based_sampling_includes_endpoints() {
        let path = SectionPath::endpoints(
            GeoPoint::new(34.05, -118.24).unwrap(),
            GeoPoint::new(33.45, -112.07).unwrap(),
        )
        .unwrap();
        let sampled = path.sample_spacing_km(100.0).unwrap();

        assert!(sampled.len() >= 2);
        assert_eq!(sampled.samples.first().unwrap().point, path.waypoints()[0]);
        assert_eq!(sampled.samples.last().unwrap().point, path.waypoints()[1]);
    }

    #[test]
    fn representative_bounds_route_uses_long_axis_midline() {
        let route = representative_route_for_bounds(
            GeoBounds::new(-109.0, -90.0, 25.0, 40.5).unwrap(),
            RepresentativeRouteStrategy::LongestAxisMidline,
        )
        .unwrap();
        assert!((route.start.lat_deg - route.end.lat_deg).abs() < 1.0e-6);
        assert!(route.start.lon_deg < route.end.lon_deg);
    }

    #[test]
    fn representative_cluster_route_uses_farthest_pair() {
        let points = [
            GeoPoint::new(37.8044, -122.2712).unwrap(),
            GeoPoint::new(38.5816, -121.4944).unwrap(),
            GeoPoint::new(39.5296, -119.8138).unwrap(),
        ];
        let route =
            representative_route_for_cluster(&points, RepresentativeRouteStrategy::FarthestPair)
                .unwrap();
        assert_eq!(route.start, points[0]);
        assert_eq!(route.end, points[2]);
    }
}
