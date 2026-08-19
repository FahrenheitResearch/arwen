use std::error::Error;

use crate::MapExtent;
use crate::features::{
    BasemapDetail, BasemapStyle, load_styled_basemap_features_for_detail,
    load_styled_basemap_polygons_for_detail,
};
use crate::presentation::LineworkRole;
use crate::projection::{ProjectionProjector, ProjectionSpec};
use crate::request::{
    Color, InverseRasterProjection, ProjectedDomain, ProjectedExtent, ProjectedLineOverlay,
    ProjectedPolygonFill,
};

const DEFAULT_BASEMAP_GRATICULE: bool = false;

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedMap {
    pub projected_x: Vec<f64>,
    pub projected_y: Vec<f64>,
    pub extent: ProjectedExtent,
    pub lines: Vec<ProjectedLineOverlay>,
    pub polygons: Vec<ProjectedPolygonFill>,
    pub inverse_raster_projection: Option<InverseRasterProjection>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProjectedBasemap {
    pub lines: Vec<ProjectedLineOverlay>,
    pub polygons: Vec<ProjectedPolygonFill>,
}

impl ProjectedMap {
    pub fn domain(&self) -> ProjectedDomain {
        ProjectedDomain {
            x: self.projected_x.clone(),
            y: self.projected_y.clone(),
            extent: self.extent.clone(),
        }
    }

    pub fn basemap(&self) -> ProjectedBasemap {
        ProjectedBasemap {
            lines: self.lines.clone(),
            polygons: self.polygons.clone(),
        }
    }

    pub fn split(self) -> (ProjectedDomain, ProjectedBasemap) {
        let domain = ProjectedDomain {
            x: self.projected_x,
            y: self.projected_y,
            extent: self.extent,
        };
        let basemap = ProjectedBasemap {
            lines: self.lines,
            polygons: self.polygons,
        };
        (domain, basemap)
    }

    pub fn rotated_degrees(mut self, degrees: f64) -> Self {
        if !degrees.is_finite() || degrees.abs() < 1.0e-9 {
            return self;
        }
        let radians = degrees.to_radians();
        let sin = radians.sin();
        let cos = radians.cos();
        let center = (
            (self.extent.x_min + self.extent.x_max) * 0.5,
            (self.extent.y_min + self.extent.y_max) * 0.5,
        );

        rotate_parallel_points(
            &mut self.projected_x,
            &mut self.projected_y,
            center,
            sin,
            cos,
        );
        for line in &mut self.lines {
            rotate_points(&mut line.points, center, sin, cos);
        }
        for polygon in &mut self.polygons {
            for ring in &mut polygon.rings {
                rotate_points(ring, center, sin, cos);
            }
        }
        self.extent = rotated_extent(self.extent, center, sin, cos);
        self.inverse_raster_projection = None;
        self
    }
}

fn rotate_parallel_points(xs: &mut [f64], ys: &mut [f64], center: (f64, f64), sin: f64, cos: f64) {
    for (x, y) in xs.iter_mut().zip(ys.iter_mut()) {
        (*x, *y) = rotate_point((*x, *y), center, sin, cos);
    }
}

fn rotate_points(points: &mut [(f64, f64)], center: (f64, f64), sin: f64, cos: f64) {
    for point in points {
        *point = rotate_point(*point, center, sin, cos);
    }
}

fn rotate_point(point: (f64, f64), center: (f64, f64), sin: f64, cos: f64) -> (f64, f64) {
    let dx = point.0 - center.0;
    let dy = point.1 - center.1;
    (
        center.0 + dx * cos - dy * sin,
        center.1 + dx * sin + dy * cos,
    )
}

fn rotated_extent(
    extent: ProjectedExtent,
    center: (f64, f64),
    sin: f64,
    cos: f64,
) -> ProjectedExtent {
    let corners = [
        (extent.x_min, extent.y_min),
        (extent.x_max, extent.y_min),
        (extent.x_max, extent.y_max),
        (extent.x_min, extent.y_max),
    ];
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for point in corners {
        let (x, y) = rotate_point(point, center, sin, cos);
        x_min = x_min.min(x);
        x_max = x_max.max(x);
        y_min = y_min.min(y);
        y_max = y_max.max(y);
    }
    ProjectedExtent {
        x_min,
        x_max,
        y_min,
        y_max,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeographicBounds {
    pub west_deg: f64,
    pub east_deg: f64,
    pub south_deg: f64,
    pub north_deg: f64,
}

impl GeographicBounds {
    pub fn new(west_deg: f64, east_deg: f64, south_deg: f64, north_deg: f64) -> Self {
        Self {
            west_deg,
            east_deg,
            south_deg: south_deg.min(north_deg),
            north_deg: south_deg.max(north_deg),
        }
    }

    fn contains(self, lat_deg: f64, lon_deg: f64) -> bool {
        if !lat_deg.is_finite() || !lon_deg.is_finite() {
            return false;
        }
        if lat_deg < self.south_deg || lat_deg > self.north_deg {
            return false;
        }
        if self.longitude_span_deg() >= 359.0 {
            return true;
        }
        let west = normalize_longitude_deg(self.west_deg);
        let east = normalize_longitude_deg(self.east_deg);
        let lon = normalize_longitude_deg(lon_deg);
        if west <= east {
            lon >= west && lon <= east
        } else {
            lon >= west || lon <= east
        }
    }

    fn center_longitude(self) -> f64 {
        if self.longitude_span_deg() >= 359.0 {
            return 0.0;
        }
        let west = normalize_longitude_deg(self.west_deg);
        let mut east = normalize_longitude_deg(self.east_deg);
        if east < west {
            east += 360.0;
        }
        normalize_longitude_deg((west + east) / 2.0)
    }

    fn longitude_span_deg(self) -> f64 {
        let raw_span = (self.east_deg - self.west_deg).abs();
        if raw_span >= 359.0 {
            return raw_span.min(360.0);
        }

        let west = normalize_longitude_deg(self.west_deg);
        let east = normalize_longitude_deg(self.east_deg);
        if west <= east {
            east - west
        } else {
            east + 360.0 - west
        }
    }
}

impl From<(f64, f64, f64, f64)> for GeographicBounds {
    fn from(value: (f64, f64, f64, f64)) -> Self {
        Self::new(value.0, value.1, value.2, value.3)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProjectedFrameSource {
    FullDomain,
    GeographicBounds(GeographicBounds),
    GeographicGridIntersection(GeographicBounds),
}

impl ProjectedFrameSource {
    fn matches(self, lat_deg: f64, lon_deg: f64) -> bool {
        match self {
            Self::FullDomain => true,
            Self::GeographicBounds(bounds) | Self::GeographicGridIntersection(bounds) => {
                bounds.contains(lat_deg, lon_deg)
            }
        }
    }

    fn geographic_bounds(self) -> Option<GeographicBounds> {
        match self {
            Self::FullDomain => None,
            Self::GeographicBounds(bounds) | Self::GeographicGridIntersection(bounds) => {
                Some(bounds)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedDomainBuildOptions {
    pub projection: Option<ProjectionSpec>,
    /// Optional latitude of origin for projection families that benefit from a
    /// caller-provided reference latitude. When absent, the builder uses the
    /// lat/lon mesh midpoint.
    pub reference_latitude_deg: Option<f64>,
    pub frame_source: ProjectedFrameSource,
    pub target_aspect_ratio: f64,
    pub fit_to_target_aspect: bool,
    pub pad_fraction: f64,
}

impl ProjectedDomainBuildOptions {
    pub fn from_bounds(bounds: (f64, f64, f64, f64), target_aspect_ratio: f64) -> Self {
        Self {
            projection: None,
            reference_latitude_deg: None,
            frame_source: ProjectedFrameSource::GeographicBounds(bounds.into()),
            target_aspect_ratio,
            fit_to_target_aspect: true,
            pad_fraction: 0.0,
        }
    }

    pub fn full_domain(target_aspect_ratio: f64) -> Self {
        Self {
            projection: None,
            reference_latitude_deg: None,
            frame_source: ProjectedFrameSource::FullDomain,
            target_aspect_ratio,
            fit_to_target_aspect: true,
            pad_fraction: 0.0,
        }
    }

    pub fn with_projection(mut self, projection: impl Into<ProjectionSpec>) -> Self {
        self.projection = Some(projection.into());
        self
    }

    pub fn with_reference_latitude(mut self, reference_latitude_deg: f64) -> Self {
        self.reference_latitude_deg = Some(reference_latitude_deg);
        self
    }

    pub fn with_geographic_grid_intersection_frame(
        mut self,
        bounds: impl Into<GeographicBounds>,
    ) -> Self {
        self.frame_source = ProjectedFrameSource::GeographicGridIntersection(bounds.into());
        self
    }

    pub fn with_natural_frame_aspect(mut self) -> Self {
        self.fit_to_target_aspect = false;
        self
    }

    pub fn with_padding(mut self, pad_fraction: f64) -> Self {
        self.pad_fraction = pad_fraction.max(0.0);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedBasemapBuildOptions {
    pub style: BasemapStyle,
    pub detail: BasemapDetail,
    pub polygon_pad_fraction: f64,
    pub line_pad_fraction: f64,
}

impl Default for ProjectedBasemapBuildOptions {
    fn default() -> Self {
        Self {
            style: BasemapStyle::Filled,
            detail: BasemapDetail::Regional,
            polygon_pad_fraction: 0.50,
            line_pad_fraction: 0.10,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedMapBuildOptions {
    pub domain: ProjectedDomainBuildOptions,
    pub basemap: Option<ProjectedBasemapBuildOptions>,
}

impl ProjectedMapBuildOptions {
    pub fn from_bounds(bounds: (f64, f64, f64, f64), target_aspect_ratio: f64) -> Self {
        Self {
            domain: ProjectedDomainBuildOptions::from_bounds(bounds, target_aspect_ratio),
            basemap: Some(ProjectedBasemapBuildOptions::default()),
        }
    }

    pub fn full_domain(target_aspect_ratio: f64) -> Self {
        Self {
            domain: ProjectedDomainBuildOptions::full_domain(target_aspect_ratio),
            basemap: Some(ProjectedBasemapBuildOptions::default()),
        }
    }

    pub fn with_projection(mut self, projection: impl Into<ProjectionSpec>) -> Self {
        self.domain = self.domain.with_projection(projection);
        self
    }

    pub fn with_geographic_grid_intersection_frame(
        mut self,
        bounds: impl Into<GeographicBounds>,
    ) -> Self {
        self.domain = self.domain.with_geographic_grid_intersection_frame(bounds);
        self
    }

    pub fn with_natural_frame_aspect(mut self) -> Self {
        self.domain = self.domain.with_natural_frame_aspect();
        self
    }

    pub fn without_basemap(mut self) -> Self {
        self.basemap = None;
        self
    }

    pub fn with_basemap_style(mut self, style: BasemapStyle) -> Self {
        let mut basemap = self.basemap.unwrap_or_default();
        basemap.style = style;
        self.basemap = Some(basemap);
        self
    }

    pub fn with_basemap_detail(mut self, detail: BasemapDetail) -> Self {
        let mut basemap = self.basemap.unwrap_or_default();
        basemap.detail = detail;
        self.basemap = Some(basemap);
        self
    }

    pub fn with_basemap_padding(
        mut self,
        line_pad_fraction: f64,
        polygon_pad_fraction: f64,
    ) -> Self {
        let mut basemap = self.basemap.unwrap_or_default();
        basemap.line_pad_fraction = line_pad_fraction.max(0.0);
        basemap.polygon_pad_fraction = polygon_pad_fraction.max(0.0);
        self.basemap = Some(basemap);
        self
    }
}

pub fn build_projected_domain(
    lat_deg: &[f32],
    lon_deg: &[f32],
    options: &ProjectedDomainBuildOptions,
) -> Result<ProjectedDomain, Box<dyn Error>> {
    validate_lat_lon_mesh(lat_deg, lon_deg)?;
    let projector = resolved_projector(lat_deg, lon_deg, options)?;
    let (projected_x, projected_y, extent) = project_domain(
        lat_deg,
        lon_deg,
        projector,
        options.frame_source,
        options.pad_fraction,
        options.target_aspect_ratio,
        options.fit_to_target_aspect,
    )?;

    Ok(ProjectedDomain {
        x: projected_x,
        y: projected_y,
        extent,
    })
}

pub fn build_projected_map_with_options(
    lat_deg: &[f32],
    lon_deg: &[f32],
    options: &ProjectedMapBuildOptions,
) -> Result<ProjectedMap, Box<dyn Error>> {
    validate_lat_lon_mesh(lat_deg, lon_deg)?;
    let projector = resolved_projector(lat_deg, lon_deg, &options.domain)?;
    let (projected_x, projected_y, extent) = project_domain(
        lat_deg,
        lon_deg,
        projector,
        options.domain.frame_source,
        options.domain.pad_fraction,
        options.domain.target_aspect_ratio,
        options.domain.fit_to_target_aspect,
    )?;

    let basemap = options
        .basemap
        .as_ref()
        .map(|basemap| {
            build_projected_basemap(projector, &extent, options.domain.frame_source, *basemap)
        })
        .transpose()?
        .unwrap_or_default();

    Ok(ProjectedMap {
        projected_x,
        projected_y,
        extent,
        lines: basemap.lines,
        polygons: basemap.polygons,
        inverse_raster_projection: None,
    })
}

pub fn build_projected_map(
    lat_deg: &[f32],
    lon_deg: &[f32],
    bounds: (f64, f64, f64, f64),
    target_ratio: f64,
) -> Result<ProjectedMap, Box<dyn Error>> {
    build_projected_map_with_options(
        lat_deg,
        lon_deg,
        &ProjectedMapBuildOptions::from_bounds(bounds, target_ratio),
    )
}

/// Project arbitrary `(lat, lon)` degree pairs into the SAME projected
/// space [`build_projected_map_with_options`] produces from the same mesh
/// and the same options.
///
/// gpuwm addition (VENDOR.md).  Every projected overlay type in
/// [`crate::request`] -- points, lines, place labels -- takes projected
/// coordinates, and until now the only way to obtain any was to be a
/// basemap feature or a grid cell.  A radar site, a range ring, a storm
/// report and a domain-boundary box are none of those: they are places,
/// known in degrees, that have to land where the map already put that
/// place.  Nothing existing calls this, so it cannot move an existing
/// pixel; it reuses `resolved_projector` rather than rebuilding one so a
/// marker can never land in a different frame than the fill it sits on.
pub fn project_geographic_points_with_options(
    lat_deg: &[f32],
    lon_deg: &[f32],
    options: &ProjectedMapBuildOptions,
    points: &[(f64, f64)],
) -> Result<Vec<(f64, f64)>, Box<dyn Error>> {
    validate_lat_lon_mesh(lat_deg, lon_deg)?;
    let projector = resolved_projector(lat_deg, lon_deg, &options.domain)?;
    Ok(points
        .iter()
        .map(|(lat, lon)| projector.project(*lat, *lon))
        .collect())
}

fn resolved_projector(
    lat_deg: &[f32],
    lon_deg: &[f32],
    options: &ProjectedDomainBuildOptions,
) -> Result<ProjectionProjector, Box<dyn Error>> {
    let projection = options
        .projection
        .clone()
        .or_else(|| ProjectionSpec::infer_from_latlon_grid(lat_deg, lon_deg))
        .ok_or("projected map builder requires at least one finite lat/lon point")?;
    let reference_longitude_deg = match (&projection, options.frame_source) {
        (ProjectionSpec::Geographic, source) => source
            .geographic_bounds()
            .map(GeographicBounds::center_longitude),
        _ => None,
    };
    projection
        .build_projector(
            options.reference_latitude_deg,
            reference_longitude_deg,
            lat_deg,
            lon_deg,
        )
        .map_err(Into::into)
}

fn validate_lat_lon_mesh(lat_deg: &[f32], lon_deg: &[f32]) -> Result<(), Box<dyn Error>> {
    if lat_deg.len() != lon_deg.len() {
        return Err("lat/lon arrays must have the same length".into());
    }
    if lat_deg.is_empty() {
        return Err("lat/lon arrays must not be empty".into());
    }
    Ok(())
}

fn project_domain(
    lat_deg: &[f32],
    lon_deg: &[f32],
    projector: ProjectionProjector,
    frame_source: ProjectedFrameSource,
    pad_fraction: f64,
    target_aspect_ratio: f64,
    fit_to_target_aspect: bool,
) -> Result<(Vec<f64>, Vec<f64>, ProjectedExtent), Box<dyn Error>> {
    let mut projected_x = Vec::with_capacity(lat_deg.len());
    let mut projected_y = Vec::with_capacity(lat_deg.len());
    let mut full_bounds = ProjectedBounds::default();
    let mut framed_bounds = ProjectedBounds::default();

    for (&lat, &lon) in lat_deg.iter().zip(lon_deg.iter()) {
        let lat = lat as f64;
        let lon = lon as f64;
        let (x, y) = projector.project(lat, lon);
        projected_x.push(x);
        projected_y.push(y);
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        full_bounds.include(x, y);
        if frame_source.matches(lat, lon) {
            framed_bounds.include(x, y);
        }
    }

    let bounds = match frame_source {
        ProjectedFrameSource::GeographicBounds(bounds) => {
            if !framed_bounds.is_valid() {
                return Err(
                    "requested geographic bounds crop does not intersect the model grid".into(),
                );
            }
            projected_geographic_frame_bounds(projector, bounds).unwrap_or(framed_bounds)
        }
        ProjectedFrameSource::GeographicGridIntersection(_) => {
            if !framed_bounds.is_valid() {
                return Err(
                    "requested geographic bounds crop does not intersect the model grid".into(),
                );
            }
            framed_bounds
        }
        ProjectedFrameSource::FullDomain if framed_bounds.is_valid() => framed_bounds,
        ProjectedFrameSource::FullDomain => full_bounds,
    };
    if !bounds.is_valid() {
        return Err("projected extent produced no finite coordinates".into());
    }

    let padded = bounds.expanded(pad_fraction.max(0.0));
    let extent = if fit_to_target_aspect {
        MapExtent::from_bounds(
            padded.min_x,
            padded.max_x,
            padded.min_y,
            padded.max_y,
            target_aspect_ratio,
        )
    } else {
        MapExtent {
            x_min: padded.min_x,
            x_max: padded.max_x,
            y_min: padded.min_y,
            y_max: padded.max_y,
        }
    };

    Ok((
        projected_x,
        projected_y,
        ProjectedExtent {
            x_min: extent.x_min,
            x_max: extent.x_max,
            y_min: extent.y_min,
            y_max: extent.y_max,
        },
    ))
}

fn projected_geographic_frame_bounds(
    projector: ProjectionProjector,
    bounds: GeographicBounds,
) -> Option<ProjectedBounds> {
    let mut projected = ProjectedBounds::default();
    let segments = if bounds.longitude_span_deg() >= 300.0 {
        180
    } else {
        96
    };
    let west = normalize_longitude_deg(bounds.west_deg);
    let mut east = normalize_longitude_deg(bounds.east_deg);
    if bounds.longitude_span_deg() >= 359.0 {
        east = west + 360.0;
    } else if east < west {
        east += 360.0;
    }

    for step in 0..=segments {
        let t = step as f64 / segments as f64;
        let lon = normalize_longitude_deg(west + (east - west) * t);
        include_projected_point(&mut projected, projector.project(bounds.south_deg, lon));
        include_projected_point(&mut projected, projector.project(bounds.north_deg, lon));
    }
    for step in 0..=segments {
        let t = step as f64 / segments as f64;
        let lat = bounds.south_deg + (bounds.north_deg - bounds.south_deg) * t;
        include_projected_point(&mut projected, projector.project(lat, bounds.west_deg));
        include_projected_point(&mut projected, projector.project(lat, bounds.east_deg));
    }

    // Full-world and near-global frames cannot be represented safely by only
    // sampling the geographic rectangle perimeter. At the antimeridian,
    // normalized -180 and +180 can collapse onto the same projected side, and
    // projections such as Robinson reach their widest x extent around the
    // equator rather than along the north/south frame edges. Sample the
    // interior grid so the fitted projected frame is centered on the real map
    // silhouette instead of being biased toward one seam side.
    if should_sample_geographic_frame_interior(bounds) {
        let lat_segments = 72usize;
        let lon_segments = 180usize;
        for lat_step in 0..=lat_segments {
            let lat_t = lat_step as f64 / lat_segments as f64;
            let lat = bounds.south_deg + (bounds.north_deg - bounds.south_deg) * lat_t;
            for lon_step in 0..=lon_segments {
                let lon_t = lon_step as f64 / lon_segments as f64;
                let lon = normalize_longitude_deg(west + (east - west) * lon_t);
                include_projected_point(&mut projected, projector.project(lat, lon));
            }
        }
    }

    projected.is_valid().then_some(projected)
}

fn should_sample_geographic_frame_interior(bounds: GeographicBounds) -> bool {
    bounds.longitude_span_deg() >= 300.0 || (bounds.north_deg - bounds.south_deg).abs() >= 120.0
}

fn include_projected_point(bounds: &mut ProjectedBounds, point: (f64, f64)) {
    if point.0.is_finite() && point.1.is_finite() {
        bounds.include(point.0, point.1);
    }
}

fn build_projected_basemap(
    projector: ProjectionProjector,
    extent: &ProjectedExtent,
    frame_source: ProjectedFrameSource,
    options: ProjectedBasemapBuildOptions,
) -> Result<ProjectedBasemap, Box<dyn Error>> {
    let line_bbox = expanded_bbox(extent, options.line_pad_fraction.max(0.0));
    let polygon_bbox = expanded_bbox(extent, options.polygon_pad_fraction.max(0.0));
    let geographic_clip = basemap_geographic_clip(frame_source);

    let mut lines = Vec::new();
    if subtle_graticule_enabled(options.detail) {
        append_graticule_lines(
            &mut lines,
            projector,
            line_bbox,
            geographic_clip,
            options.detail,
        );
    }

    let line_densify_step_deg = basemap_line_densify_step_deg(options.detail);
    let max_projected_step = max_projected_basemap_segment_length(line_bbox);
    for layer in load_styled_basemap_features_for_detail(options.style, options.detail) {
        let color = Color::rgba(layer.color.r, layer.color.g, layer.color.b, layer.color.a);
        for line in layer.lines {
            let mut current = Vec::<(f64, f64)>::with_capacity(line.len());
            let mut previous_lonlat: Option<(f64, f64)> = None;
            let mut previous_projected: Option<(f64, f64)> = None;
            for (lon, lat) in line {
                if let Some((prev_lon, prev_lat)) = previous_lonlat {
                    let steps = densified_lonlat_segment_steps(
                        prev_lon,
                        prev_lat,
                        lon,
                        lat,
                        line_densify_step_deg,
                    );
                    for step in 1..=steps {
                        let t = step as f64 / steps as f64;
                        let point_lon = interpolate_longitude(prev_lon, lon, t);
                        let point_lat = prev_lat + (lat - prev_lat) * t;
                        push_projected_line_point(
                            &mut lines,
                            &mut current,
                            &mut previous_projected,
                            projector,
                            geographic_clip,
                            line_bbox,
                            max_projected_step,
                            point_lon,
                            point_lat,
                            color,
                            layer.width,
                            layer.role,
                        );
                    }
                } else {
                    push_projected_line_point(
                        &mut lines,
                        &mut current,
                        &mut previous_projected,
                        projector,
                        geographic_clip,
                        line_bbox,
                        max_projected_step,
                        lon,
                        lat,
                        color,
                        layer.width,
                        layer.role,
                    );
                }
                previous_lonlat = Some((lon, lat));
            }
            if current.len() >= 2 {
                lines.push(ProjectedLineOverlay {
                    points: current,
                    color,
                    width: layer.width,
                    role: layer.role,
                });
            }
        }
    }

    let mut polygons = Vec::new();
    let polygon_densify_step_deg = basemap_polygon_densify_step_deg(options.detail);
    for layer in load_styled_basemap_polygons_for_detail(options.style, options.detail) {
        let color = Color::rgba(layer.color.r, layer.color.g, layer.color.b, layer.color.a);
        for polygon in layer.polygons {
            let rings: Vec<Vec<(f64, f64)>> = polygon
                .into_iter()
                .filter(|ring| {
                    geographic_clip
                        .map(|bounds| ring.iter().any(|&(lon, lat)| bounds.contains(lat, lon)))
                        .unwrap_or(true)
                })
                .map(|ring| project_densified_ring(projector, &ring, polygon_densify_step_deg))
                .filter(|ring| ring_overlaps_bbox(ring, polygon_bbox))
                .collect();
            if !rings.is_empty() {
                polygons.push(ProjectedPolygonFill {
                    rings,
                    color,
                    role: layer.role,
                });
            }
        }
    }

    Ok(ProjectedBasemap { lines, polygons })
}

fn basemap_geographic_clip(frame_source: ProjectedFrameSource) -> Option<GeographicBounds> {
    let _ = frame_source;
    None
}

fn subtle_graticule_enabled(detail: BasemapDetail) -> bool {
    if matches!(detail, BasemapDetail::Regional) {
        return false;
    }
    std::env::var("RUSTWX_BASEMAP_GRATICULE")
        .ok()
        .map(|value| parse_basemap_graticule_flag(&value))
        .unwrap_or(DEFAULT_BASEMAP_GRATICULE)
}

fn parse_basemap_graticule_flag(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn append_graticule_lines(
    lines: &mut Vec<ProjectedLineOverlay>,
    projector: ProjectionProjector,
    bbox: (f64, f64, f64, f64),
    geographic_clip: Option<GeographicBounds>,
    detail: BasemapDetail,
) {
    let color = match detail {
        BasemapDetail::Global => Color::rgba(42, 52, 66, 30),
        BasemapDetail::Broad => Color::rgba(42, 52, 66, 24),
        BasemapDetail::Regional => Color::rgba(42, 52, 66, 0),
    };
    let step_deg = match detail {
        BasemapDetail::Global => 2.0,
        BasemapDetail::Broad => 1.0,
        BasemapDetail::Regional => 0.5,
    };
    let max_projected_step = max_projected_basemap_segment_length(bbox);

    let latitude_lines: &[f64] = match detail {
        BasemapDetail::Global => &[-60.0, -30.0, 0.0, 30.0, 60.0],
        BasemapDetail::Broad => &[-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0, 80.0],
        BasemapDetail::Regional => &[],
    };
    for &lat in latitude_lines {
        let mut current = Vec::new();
        let mut previous_projected = None;
        let mut lon = -180.0;
        while lon <= 180.0 {
            push_projected_line_point(
                lines,
                &mut current,
                &mut previous_projected,
                projector,
                geographic_clip,
                bbox,
                max_projected_step,
                lon,
                lat,
                color,
                1,
                LineworkRole::Generic,
            );
            lon += step_deg;
        }
        flush_projected_line(lines, &mut current, color, 1, LineworkRole::Generic);
    }

    let lon_step = match detail {
        BasemapDetail::Global => 30,
        BasemapDetail::Broad => 20,
        BasemapDetail::Regional => 30,
    };
    for lon in (-180..=180).step_by(lon_step) {
        let mut current = Vec::new();
        let mut previous_projected = None;
        let mut lat = -80.0;
        while lat <= 80.0 {
            push_projected_line_point(
                lines,
                &mut current,
                &mut previous_projected,
                projector,
                geographic_clip,
                bbox,
                max_projected_step,
                lon as f64,
                lat,
                color,
                1,
                LineworkRole::Generic,
            );
            lat += step_deg;
        }
        flush_projected_line(lines, &mut current, color, 1, LineworkRole::Generic);
    }
}

fn basemap_line_densify_step_deg(detail: BasemapDetail) -> f64 {
    match detail {
        BasemapDetail::Global => 1.25,
        BasemapDetail::Broad => 0.9,
        BasemapDetail::Regional => 0.65,
    }
}

fn basemap_polygon_densify_step_deg(detail: BasemapDetail) -> f64 {
    match detail {
        BasemapDetail::Global => 2.0,
        BasemapDetail::Broad => 1.5,
        BasemapDetail::Regional => 1.0,
    }
}

fn max_projected_basemap_segment_length(bbox: (f64, f64, f64, f64)) -> f64 {
    let width = (bbox.1 - bbox.0).abs();
    let height = (bbox.3 - bbox.2).abs();
    width.max(height).max(1.0) * 0.30
}

fn densified_lonlat_segment_steps(
    lon0: f64,
    lat0: f64,
    lon1: f64,
    lat1: f64,
    max_step_deg: f64,
) -> usize {
    if !lon0.is_finite()
        || !lat0.is_finite()
        || !lon1.is_finite()
        || !lat1.is_finite()
        || !max_step_deg.is_finite()
        || max_step_deg <= 0.0
    {
        return 1;
    }
    let lon_span = wrapped_longitude_delta_deg(lon0, lon1).abs();
    let lat_span = (lat1 - lat0).abs();
    (lon_span.max(lat_span) / max_step_deg).ceil().max(1.0) as usize
}

fn wrapped_longitude_delta_deg(lon0: f64, lon1: f64) -> f64 {
    let mut delta = normalize_longitude_deg(lon1) - normalize_longitude_deg(lon0);
    if delta > 180.0 {
        delta -= 360.0;
    } else if delta < -180.0 {
        delta += 360.0;
    }
    delta
}

fn interpolate_longitude(lon0: f64, lon1: f64, t: f64) -> f64 {
    normalize_longitude_deg(
        normalize_longitude_deg(lon0) + wrapped_longitude_delta_deg(lon0, lon1) * t,
    )
}

fn push_projected_line_point(
    lines: &mut Vec<ProjectedLineOverlay>,
    current: &mut Vec<(f64, f64)>,
    previous_projected: &mut Option<(f64, f64)>,
    projector: ProjectionProjector,
    geographic_clip: Option<GeographicBounds>,
    bbox: (f64, f64, f64, f64),
    max_projected_step: f64,
    lon: f64,
    lat: f64,
    color: Color,
    width: u32,
    role: LineworkRole,
) {
    if geographic_clip.is_some_and(|bounds| !bounds.contains(lat, lon)) {
        flush_projected_line(lines, current, color, width, role);
        *previous_projected = None;
        return;
    }
    let point = projector.project(lat, lon);
    if !point.0.is_finite() || !point.1.is_finite() {
        flush_projected_line(lines, current, color, width, role);
        *previous_projected = None;
        return;
    }

    if let Some(previous) = *previous_projected {
        if projected_distance(previous, point) > max_projected_step {
            flush_projected_line(lines, current, color, width, role);
            if point_in_bbox(point, bbox) {
                current.push(point);
            }
            *previous_projected = Some(point);
            return;
        }
        append_clipped_projected_segment(lines, current, previous, point, bbox, color, width, role);
    } else if point_in_bbox(point, bbox) {
        current.push(point);
    }

    if !point_in_bbox(point, bbox) {
        flush_projected_line(lines, current, color, width, role);
    }
    *previous_projected = Some(point);
}

fn flush_projected_line(
    lines: &mut Vec<ProjectedLineOverlay>,
    current: &mut Vec<(f64, f64)>,
    color: Color,
    width: u32,
    role: LineworkRole,
) {
    if current.len() >= 2 {
        lines.push(ProjectedLineOverlay {
            points: std::mem::take(current),
            color,
            width,
            role,
        });
    } else {
        current.clear();
    }
}

fn append_clipped_projected_segment(
    lines: &mut Vec<ProjectedLineOverlay>,
    current: &mut Vec<(f64, f64)>,
    previous: (f64, f64),
    point: (f64, f64),
    bbox: (f64, f64, f64, f64),
    color: Color,
    width: u32,
    role: LineworkRole,
) {
    let Some((start, end)) = clip_projected_segment_to_bbox(previous, point, bbox) else {
        if !point_in_bbox(point, bbox) {
            flush_projected_line(lines, current, color, width, role);
        }
        return;
    };

    if current.is_empty() {
        current.push(start);
    } else if current
        .last()
        .is_some_and(|&last| !projected_points_close(last, start))
    {
        flush_projected_line(lines, current, color, width, role);
        current.push(start);
    }
    if current
        .last()
        .is_none_or(|&last| !projected_points_close(last, end))
    {
        current.push(end);
    }
}

fn clip_projected_segment_to_bbox(
    p0: (f64, f64),
    p1: (f64, f64),
    bbox: (f64, f64, f64, f64),
) -> Option<((f64, f64), (f64, f64))> {
    let (x_min, x_max, y_min, y_max) = bbox;
    if x_min > x_max || y_min > y_max {
        return None;
    }

    let dx = p1.0 - p0.0;
    let dy = p1.1 - p0.1;
    let mut t0 = 0.0;
    let mut t1 = 1.0;

    for (p, q) in [
        (-dx, p0.0 - x_min),
        (dx, x_max - p0.0),
        (-dy, p0.1 - y_min),
        (dy, y_max - p0.1),
    ] {
        if p.abs() <= f64::EPSILON {
            if q < 0.0 {
                return None;
            }
            continue;
        }
        let r = q / p;
        if p < 0.0 {
            if r > t1 {
                return None;
            }
            if r > t0 {
                t0 = r;
            }
        } else {
            if r < t0 {
                return None;
            }
            if r < t1 {
                t1 = r;
            }
        }
    }

    Some((
        (p0.0 + dx * t0, p0.1 + dy * t0),
        (p0.0 + dx * t1, p0.1 + dy * t1),
    ))
}

fn projected_points_close(a: (f64, f64), b: (f64, f64)) -> bool {
    projected_distance(a, b) <= 1.0e-6
}

fn projected_distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

fn project_densified_ring(
    projector: ProjectionProjector,
    ring: &[(f64, f64)],
    max_step_deg: f64,
) -> Vec<(f64, f64)> {
    if ring.is_empty() {
        return Vec::new();
    }
    let mut projected = Vec::with_capacity(ring.len());
    let mut previous_lonlat: Option<(f64, f64)> = None;
    for &(lon, lat) in ring {
        if let Some((prev_lon, prev_lat)) = previous_lonlat {
            let steps = densified_lonlat_segment_steps(prev_lon, prev_lat, lon, lat, max_step_deg);
            for step in 1..=steps {
                let t = step as f64 / steps as f64;
                let point_lon = interpolate_longitude(prev_lon, lon, t);
                let point_lat = prev_lat + (lat - prev_lat) * t;
                let point = projector.project(point_lat, point_lon);
                if point.0.is_finite() && point.1.is_finite() {
                    projected.push(point);
                }
            }
        } else {
            let point = projector.project(lat, lon);
            if point.0.is_finite() && point.1.is_finite() {
                projected.push(point);
            }
        }
        previous_lonlat = Some((lon, lat));
    }
    projected
}

fn point_in_bbox(point: (f64, f64), bbox: (f64, f64, f64, f64)) -> bool {
    point.0 >= bbox.0 && point.0 <= bbox.1 && point.1 >= bbox.2 && point.1 <= bbox.3
}

fn expanded_bbox(extent: &ProjectedExtent, pad_fraction: f64) -> (f64, f64, f64, f64) {
    let pad_x = 0.5 * pad_fraction * (extent.x_max - extent.x_min);
    let pad_y = 0.5 * pad_fraction * (extent.y_max - extent.y_min);
    (
        extent.x_min - pad_x,
        extent.x_max + pad_x,
        extent.y_min - pad_y,
        extent.y_max + pad_y,
    )
}

fn ring_overlaps_bbox(ring: &[(f64, f64)], bbox: (f64, f64, f64, f64)) -> bool {
    let mut bounds = ProjectedBounds::default();
    for &(x, y) in ring {
        bounds.include(x, y);
    }
    bounds.is_valid()
        && !(bounds.max_x < bbox.0
            || bounds.min_x > bbox.1
            || bounds.max_y < bbox.2
            || bounds.min_y > bbox.3)
}

fn normalize_longitude_deg(lon_deg: f64) -> f64 {
    let mut lon = lon_deg % 360.0;
    if lon > 180.0 {
        lon -= 360.0;
    } else if lon <= -180.0 {
        lon += 360.0;
    }
    lon
}

#[derive(Debug, Clone, Copy)]
struct ProjectedBounds {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

impl Default for ProjectedBounds {
    fn default() -> Self {
        Self {
            min_x: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            min_y: f64::INFINITY,
            max_y: f64::NEG_INFINITY,
        }
    }
}

impl ProjectedBounds {
    fn include(&mut self, x: f64, y: f64) {
        self.min_x = self.min_x.min(x);
        self.max_x = self.max_x.max(x);
        self.min_y = self.min_y.min(y);
        self.max_y = self.max_y.max(y);
    }

    fn is_valid(self) -> bool {
        self.min_x.is_finite()
            && self.max_x.is_finite()
            && self.min_y.is_finite()
            && self.max_y.is_finite()
    }

    fn expanded(self, pad_fraction: f64) -> Self {
        let width = self.max_x - self.min_x;
        let height = self.max_y - self.min_y;
        let pad_x = width * pad_fraction / 2.0;
        let pad_y = height * pad_fraction / 2.0;
        Self {
            min_x: self.min_x - pad_x,
            max_x: self.max_x + pad_x,
            min_y: self.min_y - pad_y,
            max_y: self.max_y + pad_y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::ProjectionSpec;

    fn sample_lat_lon() -> (Vec<f32>, Vec<f32>) {
        (
            vec![35.0, 35.0, 35.0, 36.0, 36.0, 36.0],
            vec![-100.0, -99.0, -98.0, -100.0, -99.0, -98.0],
        )
    }

    #[test]
    fn projected_domain_builder_supports_full_domain_geographic_projection() {
        let (lat, lon) = sample_lat_lon();
        let domain = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::full_domain(2.0)
                .with_projection(ProjectionSpec::Geographic),
        )
        .expect("domain should build");

        assert_eq!(domain.x.len(), lat.len());
        assert_eq!(domain.y.len(), lat.len());
        assert!(domain.extent.x_min < 0.0);
        assert!(domain.extent.x_max > 0.0);
        assert!(domain.extent.y_max > domain.extent.y_min);
    }

    #[test]
    fn projected_domain_builder_respects_geographic_crop_bounds() {
        let (lat, lon) = sample_lat_lon();
        let full = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::full_domain(1.5)
                .with_projection(ProjectionSpec::Geographic),
        )
        .expect("full domain");
        let cropped = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::from_bounds((-99.25, -98.25, 35.0, 36.0), 1.5)
                .with_projection(ProjectionSpec::Geographic),
        )
        .expect("cropped domain");

        assert!(
            cropped.extent.x_max - cropped.extent.x_min < full.extent.x_max - full.extent.x_min
        );
    }

    #[test]
    fn grid_intersection_frame_uses_model_cells_not_requested_rectangle() {
        let lat = vec![0.0, 0.0, 1.0, 1.0];
        let lon = vec![0.0, 1.0, 0.2, 0.8];
        let rectangle = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::from_bounds((0.1, 0.9, 0.0, 1.0), 1.0)
                .with_projection(ProjectionSpec::Geographic),
        )
        .expect("rectangle frame");
        let grid_intersection = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::from_bounds((0.1, 0.9, 0.0, 1.0), 1.0)
                .with_projection(ProjectionSpec::Geographic)
                .with_geographic_grid_intersection_frame((0.1, 0.9, 0.0, 1.0)),
        )
        .expect("grid intersection frame");

        assert!(
            grid_intersection.extent.x_max - grid_intersection.extent.x_min
                < rectangle.extent.x_max - rectangle.extent.x_min
        );
    }

    #[test]
    fn projected_frames_do_not_geographically_clip_basemap_linework() {
        let bounds = GeographicBounds::new(-125.0, -110.0, 30.0, 49.0);

        assert!(basemap_geographic_clip(ProjectedFrameSource::GeographicBounds(bounds)).is_none());
        assert!(
            basemap_geographic_clip(ProjectedFrameSource::GeographicGridIntersection(bounds))
                .is_none()
        );
    }

    #[test]
    fn projected_segments_clip_to_bbox_edges() {
        let bbox = (0.0, 10.0, 0.0, 10.0);

        let clipped = clip_projected_segment_to_bbox((-5.0, 5.0), (5.0, 5.0), bbox)
            .expect("segment should enter bbox");
        assert!((clipped.0.0 - 0.0).abs() < 1.0e-9);
        assert!((clipped.0.1 - 5.0).abs() < 1.0e-9);
        assert!((clipped.1.0 - 5.0).abs() < 1.0e-9);
        assert!((clipped.1.1 - 5.0).abs() < 1.0e-9);

        let clipped = clip_projected_segment_to_bbox((5.0, 5.0), (15.0, 5.0), bbox)
            .expect("segment should exit bbox");
        assert!((clipped.0.0 - 5.0).abs() < 1.0e-9);
        assert!((clipped.0.1 - 5.0).abs() < 1.0e-9);
        assert!((clipped.1.0 - 10.0).abs() < 1.0e-9);
        assert!((clipped.1.1 - 5.0).abs() < 1.0e-9);

        assert!(clip_projected_segment_to_bbox((-5.0, -5.0), (-1.0, -1.0), bbox).is_none());
    }

    #[test]
    fn geographic_crop_bounds_can_cross_antimeridian() {
        let lat = vec![-20.0, -18.0, -20.0, -18.0, 0.0, 0.0, 40.0, -40.0];
        let lon = vec![176.0, 178.0, -179.0, -178.0, -60.0, 30.0, 120.0, -100.0];
        let cropped = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::from_bounds((176.0, -178.0, -22.0, -15.0), 1.5)
                .with_projection(ProjectionSpec::Geographic),
        )
        .expect("cropped antimeridian domain");

        assert!(
            cropped.extent.x_max - cropped.extent.x_min < 20.0,
            "antimeridian crop should not frame the whole globe: {:?}",
            cropped.extent
        );
    }

    #[test]
    fn global_geographic_bounds_center_on_greenwich() {
        let bounds = GeographicBounds::new(-180.0, 179.999, -90.0, 90.0);

        assert_eq!(bounds.center_longitude(), 0.0);
        assert!(bounds.contains(0.0, 180.0));
        assert!(bounds.contains(0.0, -179.75));
        assert!(bounds.contains(0.0, 0.0));
    }

    #[test]
    fn global_robinson_frame_is_centered_on_world_silhouette() {
        let mut lat = Vec::new();
        let mut lon = Vec::new();
        for row_lat in [-85.0_f32, -60.0, -30.0, 0.0, 30.0, 60.0, 85.0] {
            for col_lon in (-180..=180).step_by(30) {
                lat.push(row_lat);
                lon.push(col_lon as f32);
            }
        }

        let domain = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::from_bounds((-180.0, 180.0, -85.0, 85.0), 16.0 / 9.0)
                .with_projection(ProjectionSpec::Robinson {
                    central_meridian_deg: 0.0,
                }),
        )
        .expect("global Robinson domain should build");

        let center_x = (domain.extent.x_min + domain.extent.x_max) / 2.0;
        let width = domain.extent.x_max - domain.extent.x_min;
        assert!(
            center_x.abs() < width * 0.01,
            "global Robinson frame should be centered, got extent {:?}",
            domain.extent
        );
    }

    #[test]
    fn basemap_densification_takes_short_antimeridian_path() {
        assert_eq!(
            densified_lonlat_segment_steps(179.0, 0.0, -179.0, 0.0, 1.0),
            2
        );
        assert!((interpolate_longitude(179.0, -179.0, 0.5).abs() - 180.0).abs() < 1.0e-9);
    }

    #[test]
    fn projected_ring_densification_adds_curve_support_points() {
        let projector = ProjectionSpec::Robinson {
            central_meridian_deg: 0.0,
        }
        .build_projector(None, None, &[0.0, 10.0], &[0.0, 20.0])
        .expect("projector");
        let ring = vec![(0.0, 0.0), (20.0, 10.0)];
        let projected = project_densified_ring(projector, &ring, 2.0);

        assert!(projected.len() > ring.len());
    }

    #[test]
    fn hrrr_like_crop_outside_footprint_errors_instead_of_framing_full_domain() {
        let (lat, lon) = sample_lat_lon();
        let err = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::from_bounds((8.0, 15.0, 45.0, 52.0), 1.5)
                .with_projection(ProjectionSpec::Geographic),
        )
        .expect_err("outside HRRR-like footprint should error");

        assert!(
            err.to_string().contains("does not intersect"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rap_like_crop_outside_footprint_errors_instead_of_framing_full_domain() {
        let lat = vec![20.0, 20.0, 55.0, 55.0, 35.0, 45.0];
        let lon = vec![-135.0, -60.0, -135.0, -60.0, -100.0, -80.0];
        let err = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::from_bounds((120.0, 150.0, -40.0, -20.0), 1.5)
                .with_projection(ProjectionSpec::Geographic),
        )
        .expect_err("outside RAP-like footprint should error");

        assert!(
            err.to_string().contains("does not intersect"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn gfs_like_global_crop_inside_footprint_still_builds() {
        let lat = vec![60.0, 60.0, 30.0, 30.0, 0.0, 0.0, -30.0, -30.0];
        let lon = vec![-130.0, -70.0, -120.0, -80.0, 0.0, 90.0, 120.0, -120.0];
        let cropped = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::from_bounds((-125.0, -75.0, 25.0, 50.0), 1.5)
                .with_projection(ProjectionSpec::Geographic),
        )
        .expect("GFS-like in-footprint crop should build");

        assert!(cropped.extent.x_min < cropped.extent.x_max);
        assert!(cropped.extent.y_min < cropped.extent.y_max);
    }

    #[test]
    fn projected_map_builder_can_skip_basemap_for_reusable_domain_scaffolds() {
        let (lat, lon) = sample_lat_lon();
        let projected = build_projected_map_with_options(
            &lat,
            &lon,
            &ProjectedMapBuildOptions::full_domain(1.4)
                .with_projection(ProjectionSpec::Geographic)
                .without_basemap(),
        )
        .expect("projected map");

        assert!(projected.lines.is_empty());
        assert!(projected.polygons.is_empty());
    }

    #[test]
    fn projected_map_rotation_transforms_domain_and_basemap_together() {
        let projected = ProjectedMap {
            projected_x: vec![0.0, 2.0],
            projected_y: vec![0.0, 0.0],
            extent: ProjectedExtent {
                x_min: 0.0,
                x_max: 2.0,
                y_min: 0.0,
                y_max: 2.0,
            },
            lines: vec![ProjectedLineOverlay {
                points: vec![(0.0, 1.0), (2.0, 1.0)],
                color: Color::BLACK,
                width: 1,
                role: crate::presentation::LineworkRole::Generic,
            }],
            polygons: vec![ProjectedPolygonFill {
                rings: vec![vec![(1.0, 0.0), (2.0, 1.0)]],
                color: Color::WHITE,
                role: crate::presentation::PolygonRole::Generic,
            }],
            inverse_raster_projection: None,
        }
        .rotated_degrees(90.0);

        assert!((projected.projected_x[0] - 2.0).abs() < 1.0e-9);
        assert!((projected.projected_y[0] - 0.0).abs() < 1.0e-9);
        assert!((projected.projected_x[1] - 2.0).abs() < 1.0e-9);
        assert!((projected.projected_y[1] - 2.0).abs() < 1.0e-9);
        assert!((projected.lines[0].points[0].0 - 1.0).abs() < 1.0e-9);
        assert!((projected.lines[0].points[1].0 - 1.0).abs() < 1.0e-9);
        assert!((projected.extent.x_min - 0.0).abs() < 1.0e-9);
        assert!((projected.extent.x_max - 2.0).abs() < 1.0e-9);
        assert!((projected.extent.y_min - 0.0).abs() < 1.0e-9);
        assert!((projected.extent.y_max - 2.0).abs() < 1.0e-9);
    }

    #[test]
    fn broad_and_global_graticules_are_opt_in() {
        assert!(!DEFAULT_BASEMAP_GRATICULE);
        assert!(!parse_basemap_graticule_flag(""));
        assert!(!parse_basemap_graticule_flag("false"));
        assert!(parse_basemap_graticule_flag("true"));
        assert!(parse_basemap_graticule_flag("on"));
        assert!(!subtle_graticule_enabled(BasemapDetail::Regional));
    }

    #[test]
    fn projected_map_split_preserves_domain_and_basemap_layers() {
        let projected = ProjectedMap {
            projected_x: vec![0.0, 1.0],
            projected_y: vec![0.0, 1.0],
            extent: ProjectedExtent {
                x_min: 0.0,
                x_max: 1.0,
                y_min: 0.0,
                y_max: 1.0,
            },
            lines: vec![ProjectedLineOverlay {
                points: vec![(0.0, 0.0), (1.0, 1.0)],
                color: Color::BLACK,
                width: 2,
                role: crate::presentation::LineworkRole::Generic,
            }],
            polygons: vec![ProjectedPolygonFill {
                rings: vec![vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]],
                color: Color::WHITE,
                role: crate::presentation::PolygonRole::Generic,
            }],
            inverse_raster_projection: None,
        };

        let (domain, basemap) = projected.split();
        assert_eq!(domain.x, vec![0.0, 1.0]);
        assert_eq!(basemap.lines.len(), 1);
        assert_eq!(basemap.polygons.len(), 1);
    }
}
