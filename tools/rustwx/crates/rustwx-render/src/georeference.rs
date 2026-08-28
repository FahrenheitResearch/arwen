//! What a finished PNG maps to on the Earth.
//!
//! gpuwm addition (VENDOR.md).  Until this module existed, a rendered
//! panel published its projection nowhere: not in a manifest, not in the
//! file, not on stdout.  Anything that wanted to draw on top of a finished
//! render -- a placed-grid outline, a storm track, a tile boundary -- had
//! to recover the mapping by registering coastlines against a reference,
//! and a registration fit is not a projection.  The measured failure that
//! produced this module: a global panel fitted at 4.175 px per degree of
//! longitude against 4.38 px per degree of latitude, an aspect no flat map
//! has, because the panel was Robinson and the fit was linear.  A ring
//! drawn through that transform lands in the wrong ocean.
//!
//! Two things have to be published for the mapping to be reproducible, and
//! the reason each is needed is different:
//!
//! * **The projection with every parameter resolved.**  A
//!   [`ProjectionSpec`] is not enough on its own.  `Geographic` carries no
//!   central meridian -- the builder derives one from the data -- and
//!   `LambertConformal` carries no reference latitude, which the builder
//!   also derives.  Publishing the spec alone therefore under-determines
//!   the transform by exactly the parameters that move the map sideways.
//!   [`ResolvedProjection`] is the spec with those holes filled in.
//! * **The plot rectangle.**  The map does not fill the PNG.  Title,
//!   subtitle row, colourbar and margins take pixels out of it, and how
//!   many depends on the product's visual mode.  A consumer that assumes
//!   the map is the image is wrong by tens of pixels in each direction,
//!   which at global scale is several degrees.
//!
//! Both are recorded in [`PanelGeoReference`], together with the projected
//! extent the plot rectangle spans, so `lat/lon -> pixel` is arithmetic
//! rather than inference.
//!
//! **This is a description of a render that already happened.**  It is not
//! a request and changing it changes nothing about any image.  When a
//! panel's overlay has to land on the panel, the right tool is still
//! [`crate::project_geographic_points_with_options`], which projects
//! through the panel's own projector before the image exists; this module
//! is for consumers who hold only the PNG.

use serde::{Deserialize, Serialize};

use crate::projection::{
    AlbersEqualAreaProjection, GeographicProjection, LambertConformal, MercatorProjection,
    PolarStereographic, ProjectionProjector, ProjectionSpec, RobinsonProjection,
};
use crate::request::ProjectedExtent;

/// The schema string written into every published georeference.
pub const PANEL_GEOREFERENCE_SCHEMA: &str = "rustwx.panel-georeference/v1";

/// A projection with every parameter the builder derives already resolved.
///
/// [`ProjectionSpec`] is the *request*: `Geographic` asks for a plate
/// carree without saying where its central meridian is, and
/// `LambertConformal` names two standard parallels without saying which
/// latitude the cone was pinned to.  Both are filled in from the data at
/// build time.  This enum is the *answer*, and it round-trips: a
/// [`ResolvedProjection`] rebuilds bit-identical projector state.
///
/// `Other { template }` has no representation here on purpose --
/// `rustwx-render` cannot project it either
/// ([`ProjectionSpec::build_projector`] refuses it), so a panel drawn on
/// one does not exist and there is nothing to describe.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResolvedProjection {
    /// Plate carree.  Projected units are DEGREES, not metres.
    Geographic { central_meridian_deg: f64 },
    Robinson { central_meridian_deg: f64 },
    AlbersEqualArea {
        standard_parallel_1_deg: f64,
        standard_parallel_2_deg: f64,
        central_meridian_deg: f64,
        latitude_of_origin_deg: f64,
    },
    LambertConformal {
        standard_parallel_1_deg: f64,
        standard_parallel_2_deg: f64,
        central_meridian_deg: f64,
        /// The latitude the cone was pinned to.  Derived from the data
        /// when the caller did not name one, which is why the spec alone
        /// cannot reproduce the transform.
        reference_latitude_deg: f64,
    },
    PolarStereographic {
        true_latitude_deg: f64,
        central_meridian_deg: f64,
        south_pole_on_projection_plane: bool,
    },
    Mercator {
        latitude_of_true_scale_deg: f64,
        central_meridian_deg: f64,
    },
}

impl ResolvedProjection {
    pub(crate) fn from_projector(projector: ProjectionProjector) -> Self {
        match projector {
            ProjectionProjector::Geographic(inner) => Self::Geographic {
                central_meridian_deg: inner.central_meridian_deg(),
            },
            ProjectionProjector::Robinson(inner) => Self::Robinson {
                central_meridian_deg: inner.central_meridian_deg(),
            },
            ProjectionProjector::AlbersEqualArea(inner) => Self::AlbersEqualArea {
                standard_parallel_1_deg: inner.standard_parallel_1_deg(),
                standard_parallel_2_deg: inner.standard_parallel_2_deg(),
                central_meridian_deg: inner.central_meridian_deg(),
                latitude_of_origin_deg: inner.latitude_of_origin_deg(),
            },
            ProjectionProjector::LambertConformal(inner) => Self::LambertConformal {
                standard_parallel_1_deg: inner.spec_standard_parallel_1_deg(),
                standard_parallel_2_deg: inner.spec_standard_parallel_2_deg(),
                central_meridian_deg: inner.spec_central_meridian_deg(),
                reference_latitude_deg: inner.reference_latitude_deg(),
            },
            ProjectionProjector::PolarStereographic(inner) => Self::PolarStereographic {
                true_latitude_deg: inner.true_latitude_deg(),
                central_meridian_deg: inner.central_meridian_deg(),
                south_pole_on_projection_plane: inner.south_pole_on_projection_plane(),
            },
            ProjectionProjector::Mercator(inner) => Self::Mercator {
                latitude_of_true_scale_deg: inner.latitude_of_true_scale_deg(),
                central_meridian_deg: inner.central_meridian_deg(),
            },
        }
    }

    pub(crate) fn projector(self) -> ProjectionProjector {
        match self {
            Self::Geographic {
                central_meridian_deg,
            } => ProjectionProjector::Geographic(GeographicProjection::new(central_meridian_deg)),
            Self::Robinson {
                central_meridian_deg,
            } => ProjectionProjector::Robinson(RobinsonProjection::new(central_meridian_deg)),
            Self::AlbersEqualArea {
                standard_parallel_1_deg,
                standard_parallel_2_deg,
                central_meridian_deg,
                latitude_of_origin_deg,
            } => ProjectionProjector::AlbersEqualArea(AlbersEqualAreaProjection::new(
                standard_parallel_1_deg,
                standard_parallel_2_deg,
                central_meridian_deg,
                latitude_of_origin_deg,
            )),
            Self::LambertConformal {
                standard_parallel_1_deg,
                standard_parallel_2_deg,
                central_meridian_deg,
                reference_latitude_deg,
            } => ProjectionProjector::LambertConformal(LambertConformal::new(
                standard_parallel_1_deg,
                standard_parallel_2_deg,
                central_meridian_deg,
                reference_latitude_deg,
            )),
            Self::PolarStereographic {
                true_latitude_deg,
                central_meridian_deg,
                south_pole_on_projection_plane,
            } => ProjectionProjector::PolarStereographic(PolarStereographic::new(
                true_latitude_deg,
                central_meridian_deg,
                south_pole_on_projection_plane,
            )),
            Self::Mercator {
                latitude_of_true_scale_deg,
                central_meridian_deg,
            } => ProjectionProjector::Mercator(MercatorProjection::new(
                latitude_of_true_scale_deg,
                central_meridian_deg,
            )),
        }
    }

    /// `(lat, lon)` degrees -> projected `(x, y)`.
    ///
    /// Units are metres for every variant except `Geographic`, which is
    /// degrees.  That asymmetry is the renderer's, not this module's --
    /// it is inherited so the published extent and the published
    /// projection are in the same units as each other.
    pub fn project(self, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        self.projector().project(lat_deg, lon_deg)
    }

    /// Projected `(x, y)` -> `(lat, lon)` degrees.
    ///
    /// `None` where the projection has no inverse at that point.  Polar
    /// stereographic has no inverse in this crate at all and always
    /// returns `None`; a consumer that needs pixel->earth on one has to
    /// say so rather than get a plausible wrong answer.
    pub fn unproject(self, x: f64, y: f64) -> Option<(f64, f64)> {
        self.projector().unproject(x, y)
    }

    /// The request this was resolved from, for callers that only need the
    /// family.  Lossy in the direction that matters: the resolved
    /// parameters are dropped.
    pub fn spec(self) -> ProjectionSpec {
        match self {
            Self::Geographic { .. } => ProjectionSpec::Geographic,
            Self::Robinson {
                central_meridian_deg,
            } => ProjectionSpec::Robinson {
                central_meridian_deg,
            },
            Self::AlbersEqualArea {
                standard_parallel_1_deg,
                standard_parallel_2_deg,
                central_meridian_deg,
                latitude_of_origin_deg,
            } => ProjectionSpec::AlbersEqualArea {
                standard_parallel_1_deg,
                standard_parallel_2_deg,
                central_meridian_deg,
                latitude_of_origin_deg,
            },
            Self::LambertConformal {
                standard_parallel_1_deg,
                standard_parallel_2_deg,
                central_meridian_deg,
                ..
            } => ProjectionSpec::LambertConformal {
                standard_parallel_1_deg,
                standard_parallel_2_deg,
                central_meridian_deg,
            },
            Self::PolarStereographic {
                true_latitude_deg,
                central_meridian_deg,
                south_pole_on_projection_plane,
            } => ProjectionSpec::PolarStereographic {
                true_latitude_deg,
                central_meridian_deg,
                south_pole_on_projection_plane,
            },
            Self::Mercator {
                latitude_of_true_scale_deg,
                central_meridian_deg,
            } => ProjectionSpec::Mercator {
                latitude_of_true_scale_deg,
                central_meridian_deg,
            },
        }
    }
}

/// The map's pixel box inside the PNG: `x`, `y` of its top-left corner,
/// then `width` and `height`.
///
/// The map is NOT the image.  A 1600x1200 panel with a title row and a
/// vertical colourbar puts its map in roughly 1450x1130 of those pixels,
/// offset down and right.  Assuming otherwise is an error of several
/// degrees on a global frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlotRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Everything needed to put a latitude and longitude on a pixel of a
/// finished panel, and to read one back off.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PanelGeoReference {
    pub schema: String,
    /// The PNG's own size.
    pub image_width_px: u32,
    pub image_height_px: u32,
    /// Where the map sits inside it.
    pub plot_rect_px: PlotRect,
    /// The projection, with every derived parameter resolved.
    pub projection: ResolvedProjection,
    /// The projected-coordinate box the plot rectangle spans.  Units
    /// follow the projection: metres, except degrees for `Geographic`.
    pub extent: ProjectedExtent,
    /// `(west, east, south, north)` of the DATA, in degrees.  Carried for
    /// sanity checks and captions; the transform does not use it, and it
    /// is not the same box as `extent` because the frame is padded and
    /// reshaped to the panel's aspect ratio.
    pub geographic_bounds: (f64, f64, f64, f64),
}

impl PanelGeoReference {
    pub fn new(
        image_width_px: u32,
        image_height_px: u32,
        plot_rect_px: PlotRect,
        projection: ResolvedProjection,
        extent: ProjectedExtent,
        geographic_bounds: (f64, f64, f64, f64),
    ) -> Self {
        Self {
            schema: PANEL_GEOREFERENCE_SCHEMA.to_string(),
            image_width_px,
            image_height_px,
            plot_rect_px,
            projection,
            extent,
            geographic_bounds,
        }
    }

    /// `(lat, lon)` degrees -> `(x, y)` pixel in the FULL image.
    ///
    /// `None` when the point projects outside the plot rectangle by more
    /// than a tenth of its span, which is the same tolerance
    /// [`crate::MapExtent::to_pixel`] applies when it draws.  Matching
    /// that tolerance is deliberate: a consumer asking "where would the
    /// renderer have put this?" gets the renderer's own answer, including
    /// its answer of "off the map".
    pub fn lonlat_to_pixel(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        let (x, y) = self.projection.project(lat_deg, lon_deg);
        self.projected_to_pixel(x, y)
    }

    /// Projected `(x, y)` -> `(x, y)` pixel in the FULL image.
    pub fn projected_to_pixel(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let dx = self.extent.x_max - self.extent.x_min;
        let dy = self.extent.y_max - self.extent.y_min;
        if dx.abs() < 1.0e-12 || dy.abs() < 1.0e-12 {
            return None;
        }
        let rx = (x - self.extent.x_min) / dx;
        let ry = 1.0 - (y - self.extent.y_min) / dy;
        if !(-0.1..=1.1).contains(&rx) || !(-0.1..=1.1).contains(&ry) {
            return None;
        }
        let width = self.plot_rect_px.width.saturating_sub(1) as f64;
        let height = self.plot_rect_px.height.saturating_sub(1) as f64;
        Some((
            self.plot_rect_px.x as f64 + rx * width,
            self.plot_rect_px.y as f64 + ry * height,
        ))
    }

    /// `(x, y)` pixel in the FULL image -> `(lat, lon)` degrees.
    ///
    /// `None` outside the plot rectangle, and `None` where the projection
    /// has no inverse.  A pixel inside the rectangle but outside the
    /// world silhouette -- the corners of a Robinson frame are ocean-free
    /// background, not sea -- also returns `None`.
    pub fn pixel_to_lonlat(&self, x_px: f64, y_px: f64) -> Option<(f64, f64)> {
        let width = self.plot_rect_px.width.saturating_sub(1) as f64;
        let height = self.plot_rect_px.height.saturating_sub(1) as f64;
        if width <= 0.0 || height <= 0.0 {
            return None;
        }
        let rx = (x_px - self.plot_rect_px.x as f64) / width;
        let ry = (y_px - self.plot_rect_px.y as f64) / height;
        if !(0.0..=1.0).contains(&rx) || !(0.0..=1.0).contains(&ry) {
            return None;
        }
        let x = self.extent.x_min + rx * (self.extent.x_max - self.extent.x_min);
        let y = self.extent.y_min + (1.0 - ry) * (self.extent.y_max - self.extent.y_min);
        self.projection.unproject(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn georeference(projection: ResolvedProjection, extent: ProjectedExtent) -> PanelGeoReference {
        PanelGeoReference::new(
            1600,
            1200,
            PlotRect {
                x: 18,
                y: 46,
                width: 1451,
                height: 1124,
            },
            projection,
            extent,
            (-180.0, 180.0, -90.0, 90.0),
        )
    }

    /// The defect this module exists to prevent, stated as a measurement.
    ///
    /// A global panel is Robinson.  Robinson's longitude scale shrinks
    /// with latitude, so no single pixels-per-degree fits it.  This test
    /// measures the fitted scale at the equator against the fitted scale
    /// at 60 degrees and asserts they disagree by more than a third --
    /// which is exactly why the linear registration that produced
    /// "4.175 px/deg longitude against 4.38 px/deg latitude" could not
    /// have worked, whatever it fitted against.
    #[test]
    fn a_global_panel_has_no_single_pixels_per_degree() {
        let projection = ResolvedProjection::Robinson {
            central_meridian_deg: 0.0,
        };
        let (equator_x, _) = projection.project(0.0, 10.0);
        let (sixty_x, _) = projection.project(60.0, 10.0);
        let ratio = sixty_x / equator_x;
        // Measured: 0.7986.  One degree of longitude is 20 per cent
        // narrower at 60 degrees than at the equator, so a single
        // pixels-per-degree number describes neither.
        assert!(
            ratio < 0.85,
            "Robinson must compress longitude towards the poles; got ratio {ratio}"
        );
    }

    #[test]
    fn a_point_round_trips_through_the_published_transform() {
        let projection = ResolvedProjection::Robinson {
            central_meridian_deg: 0.0,
        };
        // The world silhouette's bounding box, which is what the renderer
        // frames a global panel to.  Robinson is widest at the equator and
        // tallest at the poles, so the two extremes come from different
        // points -- taking both from one corner would give a box narrower
        // than the map that gets drawn.  Just inside the antimeridian,
        // because the renderer's longitude normalisation folds an exact
        // -180 onto +180.
        let (x_max, _) = projection.project(0.0, 179.99);
        let x_min = -x_max;
        let (_, y_max) = projection.project(90.0, 0.0);
        let y_min = -y_max;
        let georeference = georeference(
            projection,
            ProjectedExtent {
                x_min,
                x_max,
                y_min,
                y_max,
            },
        );
        // The Southern Ocean cyclone the cascade placed a grid on.
        let (lat, lon) = (-60.079, 139.499);
        let (px, py) = georeference
            .lonlat_to_pixel(lat, lon)
            .expect("a point inside the frame must land on a pixel");
        let (back_lat, back_lon) = georeference
            .pixel_to_lonlat(px, py)
            .expect("a pixel inside the plot rectangle must read back");
        assert!(
            (back_lat - lat).abs() < 0.2 && (back_lon - lon).abs() < 0.2,
            "round trip moved the point: {lat},{lon} -> {px},{py} -> {back_lat},{back_lon}"
        );
    }

    /// The plot rectangle is not the image, and the difference is not
    /// small: at 1600x1200 the map starts 18 px right and 46 px down.
    #[test]
    fn the_plot_rectangle_offsets_every_pixel() {
        let projection = ResolvedProjection::Geographic {
            central_meridian_deg: 0.0,
        };
        let georeference = georeference(
            projection,
            ProjectedExtent {
                x_min: -180.0,
                x_max: 180.0,
                y_min: -90.0,
                y_max: 90.0,
            },
        );
        let (px, py) = georeference
            .lonlat_to_pixel(90.0, -179.999_999)
            .expect("the north-west corner projects");
        assert!(
            (px - 18.0).abs() < 0.01 && (py - 46.0).abs() < 0.01,
            "the north-west corner of the data must land on the plot rect origin, got {px},{py}"
        );
    }

    #[test]
    fn every_variant_rebuilds_its_own_projector() {
        let cases = [
            ResolvedProjection::Geographic {
                central_meridian_deg: -96.0,
            },
            ResolvedProjection::Robinson {
                central_meridian_deg: 12.0,
            },
            ResolvedProjection::AlbersEqualArea {
                standard_parallel_1_deg: 29.5,
                standard_parallel_2_deg: 45.5,
                central_meridian_deg: -96.0,
                latitude_of_origin_deg: 37.5,
            },
            ResolvedProjection::LambertConformal {
                standard_parallel_1_deg: 33.0,
                standard_parallel_2_deg: 45.0,
                central_meridian_deg: -96.0,
                reference_latitude_deg: 39.0,
            },
            ResolvedProjection::PolarStereographic {
                true_latitude_deg: 60.0,
                central_meridian_deg: -105.0,
                south_pole_on_projection_plane: false,
            },
            ResolvedProjection::Mercator {
                latitude_of_true_scale_deg: 0.0,
                central_meridian_deg: 0.0,
            },
        ];
        for case in cases {
            // The round trip that matters: the published parameters have
            // to rebuild the projector, and the rebuilt projector has to
            // agree bit for bit.  (JSON is exercised where the manifest is
            // actually written; this crate carries no JSON dependency.)
            let rebuilt = ResolvedProjection::from_projector(case.projector());
            assert_eq!(case, rebuilt, "rebuilding changed {case:?}");
            assert_eq!(
                case.project(35.0, -90.0),
                rebuilt.project(35.0, -90.0),
                "a rebuilt projection must project identically"
            );
        }
    }

    /// A Lambert panel published without its reference latitude cannot be
    /// reproduced.  This is the concrete reason `ResolvedProjection` is a
    /// separate type from `ProjectionSpec` rather than a rename of it.
    #[test]
    fn the_reference_latitude_moves_the_map() {
        let pinned_at_39 = ResolvedProjection::LambertConformal {
            standard_parallel_1_deg: 33.0,
            standard_parallel_2_deg: 45.0,
            central_meridian_deg: -96.0,
            reference_latitude_deg: 39.0,
        };
        let pinned_at_45 = ResolvedProjection::LambertConformal {
            standard_parallel_1_deg: 33.0,
            standard_parallel_2_deg: 45.0,
            central_meridian_deg: -96.0,
            reference_latitude_deg: 45.0,
        };
        let (_, y_39) = pinned_at_39.project(35.0, -90.0);
        let (_, y_45) = pinned_at_45.project(35.0, -90.0);
        assert!(
            (y_39 - y_45).abs() > 100_000.0,
            "six degrees of reference latitude must move the map by hundreds of km; got {}",
            (y_39 - y_45).abs()
        );
    }

    #[test]
    fn a_pixel_outside_the_plot_rectangle_is_refused() {
        let georeference = georeference(
            ResolvedProjection::Geographic {
                central_meridian_deg: 0.0,
            },
            ProjectedExtent {
                x_min: -180.0,
                x_max: 180.0,
                y_min: -90.0,
                y_max: 90.0,
            },
        );
        // Inside the image, above the map: the title row.
        assert!(georeference.pixel_to_lonlat(800.0, 10.0).is_none());
        // Inside the image, right of the map: the colourbar.
        assert!(georeference.pixel_to_lonlat(1550.0, 600.0).is_none());
        // Inside the map.
        assert!(georeference.pixel_to_lonlat(800.0, 600.0).is_some());
    }
}
