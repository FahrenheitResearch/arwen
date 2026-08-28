pub mod advisory;
mod color;
mod colorbar;
mod colormap;
mod colormaps;
mod contour_fill;
mod draw;
mod error;
mod features;
pub mod georeference;
mod overlay;
mod panel;
mod presentation;
mod projected_map;
mod projection;
mod rasterize;
mod render;
mod request;
mod text;
pub mod weather;

pub use contour_fill::{
    ProjectedContourGeometry, ProjectedContourGeometryTiming, ProjectedContourLineStyle,
    build_projected_contour_geometry, build_projected_contour_geometry_profile,
};
pub use error::RustwxRenderError;
pub use rasterize::{cuda_rasterize_stats, print_cuda_rasterize_stats_if_enabled};

/// Always a no-op in this port (the upstream cuda feature was stripped);
/// kept so downstream callers compile unchanged.
pub fn print_cuda_rasterize_phase_timing_if_enabled() {}
pub use features::{
    BasemapDetail, BasemapStyle, StyledLonLatLayer, StyledLonLatPolygonLayer,
    checked_in_natural_earth_110m_root, load_styled_basemap_features,
    load_styled_basemap_features_for, load_styled_basemap_features_for_detail,
    load_styled_basemap_polygons, load_styled_basemap_polygons_for, load_styled_conus_features_for,
    load_styled_conus_polygons_for,
};
pub use image::RgbaImage;
pub use panel::{PanelGridLayout, PanelPadding, compose_panel_images, render_panel_grid};
pub use presentation::{
    LineworkRole, PolygonRole, ProductVisualMode, RenderPresentation, StaticPlotStyle,
};
pub use projected_map::{
    resolved_projection_for_options,
    GeographicBounds, ProjectedBasemap, ProjectedBasemapBuildOptions, ProjectedDomainBuildOptions,
    ProjectedFrameSource, ProjectedMap, ProjectedMapBuildOptions, build_projected_domain,
    build_projected_map, build_projected_map_with_options,
    project_geographic_points_with_options,
};
pub use georeference::{
    PANEL_GEOREFERENCE_SCHEMA, PanelGeoReference, PlotRect, ResolvedProjection,
};
pub use projection::{LambertConformal, ProjectionSpec};
pub use render::{
    PngCompressionMode, PngWriteOptions, RenderImageTiming, RenderPngTiming,
    map_frame_aspect_ratio, map_frame_aspect_ratio_for_mode,
    map_frame_aspect_ratio_for_mode_with_chrome_scale,
    map_frame_aspect_ratio_for_mode_with_domain_frame,
    map_frame_aspect_ratio_for_mode_with_domain_frame_and_chrome_scale, render_to_image_profile,
    render_to_png_profile as profile_render_to_png,
};
pub use request::{
    ChromeScale, Color, ColorScale, ContourLayer, ContourLinePattern, ContourStyle,
    DiscreteColorScale, DomainFrame, DomainFrameSource, ExtendMode, Field2D, GeographicClipBounds,
    GridShape, InverseRasterProjection, LatLonGrid, MapRenderRequest, ProductKey, ProductMaturity,
    ProductSemanticFlag, ProductSemantics, ProjectedDomain, ProjectedExtent,
    ProjectedLabelPlacement, ProjectedLineOverlay, ProjectedMarkerShape, ProjectedPlaceLabel,
    ProjectedPlaceLabelPriority, ProjectedPlaceLabelStyle, ProjectedPointOverlay,
    ProjectedPolygonFill, RasterSampleMode, RgbaGridField, WindBarbLayer, WindBarbStyle,
    WindStreamlineLayer, WindStreamlineStyle,
};
pub use rustwx_core::{
    Field2D as CoreField2D, GridProjection as CoreGridProjection, GridShape as CoreGridShape,
    LatLonGrid as CoreLatLonGrid, ProductKey as CoreProductKey,
};
pub use weather::{
    DerivedProductStyle, DerivedScalePreset, ECAPE_SEVERE_PANEL_PRODUCTS,
    SEVERE_CLASSIC_PANEL_PRODUCTS, WeatherPalette, WeatherPreset, WeatherProduct, palette_scale,
};

pub use crate::color::Rgba;
pub use crate::colorbar::{legend_color_at_rel, legend_tick_rel};
use crate::colormap::Extend;
pub use crate::colormap::{
    ColormapBuildOptions, LegendControls, LegendMode, LevelDensity, LeveledColormap, RenderDensity,
    densify_discrete_scale,
};
use crate::overlay::{
    BarbOverlay, ContourOverlay, InverseProjectedGrid, MapExtent, ProjectedGrid,
    ProjectedPlaceLabelOverlay, ProjectedPointOverlay as RenderProjectedPointOverlay,
    ProjectedPolygon, ProjectedPolyline,
};
use crate::render::{
    RenderOpts, center_horizontal_canvas_content, crop_canvas_whitespace,
    encode_rgba_png_profile_with_options, render_to_image as native_render_to_image, render_to_png,
    trim_vertical_canvas_whitespace,
};
pub use crate::text::format_tick;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

fn trim_vertical_canvas_whitespace_enabled() -> bool {
    std::env::var("RUSTWX_TRIM_VERTICAL_WHITESPACE")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RustRenderer;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RenderStateTiming {
    pub validate_ms: u128,
    pub data_buffer_ms: u128,
    pub projected_grid_ms: u128,
    pub projected_lines_ms: u128,
    pub projected_polygons_ms: u128,
    pub contour_prep_ms: u128,
    pub barb_prep_ms: u128,
    pub state_prep_ms: u128,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RenderSaveTiming {
    pub state_timing: RenderStateTiming,
    pub png_timing: RenderPngTiming,
    pub file_write_ms: u128,
    pub total_ms: u128,
    /// What the PNG that was just written maps to on the Earth.
    ///
    /// gpuwm addition (VENDOR.md).  `Some` only when every ingredient is
    /// present AND still true of the written bytes: the caller published a
    /// resolved projection and its geographic bounds on the request, the
    /// request carried a projected domain (its extent is the projected
    /// box), and no post-render pass moved the map inside the canvas.
    /// Anything less and this is `None` with the reason below, because a
    /// georeference that is tens of pixels wrong is the defect this exists
    /// to fix, not a lesser version of the fix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub georeference: Option<georeference::PanelGeoReference>,
    /// Why `georeference` is `None`, naming the missing ingredient.  A
    /// manifest that lists a panel without a transform must say what was
    /// missing -- a silent omission is exactly the defect being fixed.
    /// `None` whenever `georeference` is `Some`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub georeference_absent_reason: Option<String>,
}

/// The publish decision for one finished panel: the georeference when the
/// request supplied every ingredient and the written PNG still matches the
/// layout, otherwise the reason it is withheld (gpuwm addition, VENDOR.md).
fn panel_georeference_for_save(
    request: &MapRenderRequest,
    image_timing: &RenderImageTiming,
) -> (
    Option<georeference::PanelGeoReference>,
    Option<String>,
) {
    let Some(resolved_projection) = request.resolved_projection else {
        return (
            None,
            Some(
                "the render request carried no resolved projection; the caller did not \
                 publish which projection the projected domain was built in"
                    .to_string(),
            ),
        );
    };
    let Some(projected_domain) = request.projected_domain.as_ref() else {
        return (
            None,
            Some(
                "the render request carried no projected domain, so no projected extent \
                 exists to map pixels through"
                    .to_string(),
            ),
        );
    };
    let Some(geographic_bounds) = request.geographic_bounds else {
        return (
            None,
            Some(
                "the render request carried no geographic bounds; publishing a transform \
                 with fabricated bounds would be worse than publishing none"
                    .to_string(),
            ),
        );
    };
    if !image_timing.plot_rect_describes_the_png {
        return (
            None,
            Some(
                "a post-render pass (map-viewport crop, horizontal recentre, or vertical \
                 trim) moved the map in a way its reported offset cannot describe -- the \
                 adjusted plot rectangle would fall outside the written PNG"
                    .to_string(),
            ),
        );
    }
    if image_timing.map_w == 0 || image_timing.map_h == 0 {
        return (
            None,
            Some("the layout reported a zero-size map rectangle".to_string()),
        );
    }
    // The written PNG's own size: the post-render passes can shrink the
    // canvas below the requested size, and a georeference must describe
    // the FILE, not the request.  The request dims are only a fallback
    // for a timing that predates the recorded size.
    let (image_width_px, image_height_px) =
        if image_timing.image_w > 0 && image_timing.image_h > 0 {
            (image_timing.image_w, image_timing.image_h)
        } else {
            (request.width, request.height)
        };
    (
        Some(georeference::PanelGeoReference::new(
            image_width_px,
            image_height_px,
            georeference::PlotRect {
                x: image_timing.map_x,
                y: image_timing.map_y,
                width: image_timing.map_w,
                height: image_timing.map_h,
            },
            resolved_projection,
            projected_domain.extent.clone(),
            geographic_bounds,
        )),
        None,
    )
}

#[derive(Default)]
struct RenderScratch {
    f64_buffers: Vec<Vec<f64>>,
    point_buffers: Vec<Vec<(f64, f64)>>,
}

impl RenderScratch {
    fn take_f64_buffer(&mut self, len: usize) -> Vec<f64> {
        let mut buffer = self.f64_buffers.pop().unwrap_or_default();
        buffer.clear();
        if buffer.capacity() < len {
            buffer.reserve(len - buffer.capacity());
        }
        buffer
    }

    fn fill_f64_from_f32(&mut self, src: &[f32]) -> Vec<f64> {
        let mut buffer = self.take_f64_buffer(src.len());
        buffer.extend(src.iter().map(|&value| value as f64));
        buffer
    }

    fn fill_f64_from_f64(&mut self, src: &[f64]) -> Vec<f64> {
        let mut buffer = self.take_f64_buffer(src.len());
        buffer.extend_from_slice(src);
        buffer
    }

    fn fill_f64_constant(&mut self, len: usize, value: f64) -> Vec<f64> {
        let mut buffer = self.take_f64_buffer(len);
        buffer.resize(len, value);
        buffer
    }

    fn reclaim_f64_buffer(&mut self, mut buffer: Vec<f64>) {
        buffer.clear();
        self.f64_buffers.push(buffer);
    }

    fn take_point_buffer(&mut self, len: usize) -> Vec<(f64, f64)> {
        let mut buffer = self.point_buffers.pop().unwrap_or_default();
        buffer.clear();
        if buffer.capacity() < len {
            buffer.reserve(len - buffer.capacity());
        }
        buffer
    }

    fn fill_point_buffer(&mut self, src: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let mut buffer = self.take_point_buffer(src.len());
        buffer.extend_from_slice(src);
        buffer
    }

    fn reclaim_point_buffer(&mut self, mut buffer: Vec<(f64, f64)>) {
        buffer.clear();
        self.point_buffers.push(buffer);
    }

    fn reclaim_render_opts(&mut self, mut opts: RenderOpts, data: Vec<f64>) {
        self.reclaim_f64_buffer(data);

        if let Some(grid) = opts.projected_grid.take() {
            self.reclaim_f64_buffer(grid.x);
            self.reclaim_f64_buffer(grid.y);
        }
        if let Some(grid) = opts.inverse_projected_grid.take() {
            self.reclaim_f64_buffer(grid.lat_deg);
            self.reclaim_f64_buffer(grid.lon_deg);
        }

        for line in opts.projected_lines.drain(..) {
            self.reclaim_point_buffer(line.points);
        }

        for poly in opts.projected_polygons.drain(..) {
            for ring in poly.rings {
                self.reclaim_point_buffer(ring);
            }
        }

        for poly in opts.projected_data_polygons.drain(..) {
            for ring in poly.rings {
                self.reclaim_point_buffer(ring);
            }
        }

        for contour in opts.contours.drain(..) {
            self.reclaim_f64_buffer(contour.data);
            self.reclaim_f64_buffer(contour.levels);
        }

        opts.barbs.clear();

        for streamline in opts.streamlines.drain(..) {
            self.reclaim_f64_buffer(streamline.u);
            self.reclaim_f64_buffer(streamline.v);
        }
    }
}

thread_local! {
    static RENDER_SCRATCH: RefCell<RenderScratch> = RefCell::new(RenderScratch::default());
}

impl RustRenderer {
    pub fn render_png(self, request: &MapRenderRequest) -> Result<Vec<u8>, RustwxRenderError> {
        with_render_state(request, |data, ny, nx, opts| {
            Ok(render_to_png(data, ny, nx, opts))
        })
    }

    pub fn render_image(self, request: &MapRenderRequest) -> Result<RgbaImage, RustwxRenderError> {
        with_render_state(request, |data, ny, nx, opts| {
            Ok(native_render_to_image(data, ny, nx, opts))
        })
    }

    pub fn render_image_with_style(
        self,
        request: &MapRenderRequest,
        plot_style: StaticPlotStyle,
    ) -> Result<RgbaImage, RustwxRenderError> {
        with_render_state_with_style(request, plot_style, |data, ny, nx, opts| {
            Ok(native_render_to_image(data, ny, nx, opts))
        })
    }

    pub fn save_png<P: AsRef<Path>>(
        self,
        request: &MapRenderRequest,
        output_path: P,
    ) -> Result<(), RustwxRenderError> {
        self.save_png_profile_with_options(request, output_path, &PngWriteOptions::default())
            .map(|_| ())
    }

    pub fn save_png_profile<P: AsRef<Path>>(
        self,
        request: &MapRenderRequest,
        output_path: P,
    ) -> Result<RenderSaveTiming, RustwxRenderError> {
        self.save_png_profile_with_options(request, output_path, &PngWriteOptions::default())
    }

    pub fn save_png_profile_with_options<P: AsRef<Path>>(
        self,
        request: &MapRenderRequest,
        output_path: P,
        png_options: &PngWriteOptions,
    ) -> Result<RenderSaveTiming, RustwxRenderError> {
        self.save_png_profile_with_options_and_style(
            request,
            output_path,
            png_options,
            StaticPlotStyle::from_env(),
        )
    }

    pub fn save_png_profile_with_options_and_style<P: AsRef<Path>>(
        self,
        request: &MapRenderRequest,
        output_path: P,
        png_options: &PngWriteOptions,
        plot_style: StaticPlotStyle,
    ) -> Result<RenderSaveTiming, RustwxRenderError> {
        let total_start = Instant::now();
        let (bytes, state_timing, png_timing) =
            with_render_state_profile_with_style(request, plot_style, |data, ny, nx, opts| {
                let (image, mut image_timing) = render_to_image_profile(data, ny, nx, opts);
                let trim_start = Instant::now();
                // gpuwm addition (VENDOR.md): each of these three passes
                // can move the map inside the canvas.  They used to
                // report nothing, so the plot rectangle died with the
                // first pass -- and since every REGIONAL panel's domain
                // frame takes one, only global panels ever published a
                // georeference, which is not "fixed" by default.  Each
                // pass now reports the offset it applied; the rectangle
                // FOLLOWS the map, and it is retired only when the
                // adjusted rectangle would fall outside the written image
                // (a pass cut into the map itself), because publishing
                // that rectangle would be quietly wrong.
                let mut moved_x: i64 = 0;
                let mut moved_y: i64 = 0;
                let image = match opts.domain_frame {
                    Some(frame) if matches!(frame.source, DomainFrameSource::MapViewport) => {
                        let (cropped, (crop_left, crop_top)) = crop_canvas_whitespace(
                            &image,
                            opts.presentation.canvas_background,
                            8,
                        );
                        moved_x -= i64::from(crop_left);
                        moved_y -= i64::from(crop_top);
                        cropped
                    }
                    Some(_) => {
                        let (centered, shift_x) = center_horizontal_canvas_content(
                            &image,
                            opts.presentation.canvas_background,
                        );
                        moved_x += shift_x;
                        centered
                    }
                    None => image,
                };
                let trimmed = if trim_vertical_canvas_whitespace_enabled() {
                    let (trimmed, crop_top) = trim_vertical_canvas_whitespace(
                        &image,
                        opts.presentation.canvas_background,
                    );
                    moved_y -= i64::from(crop_top);
                    trimmed
                } else {
                    image
                };
                image_timing.postprocess_offset_x = moved_x;
                image_timing.postprocess_offset_y = moved_y;
                image_timing.image_w = trimmed.width();
                image_timing.image_h = trimmed.height();
                let adjusted_x = i64::from(image_timing.map_x) + moved_x;
                let adjusted_y = i64::from(image_timing.map_y) + moved_y;
                let rect_fits = adjusted_x >= 0
                    && adjusted_y >= 0
                    && adjusted_x + i64::from(image_timing.map_w) <= i64::from(trimmed.width())
                    && adjusted_y + i64::from(image_timing.map_h) <= i64::from(trimmed.height());
                if rect_fits {
                    image_timing.map_x = adjusted_x as u32;
                    image_timing.map_y = adjusted_y as u32;
                } else {
                    image_timing.plot_rect_describes_the_png = false;
                }
                let trim_ms = trim_start.elapsed().as_millis();
                image_timing.postprocess_ms = image_timing.postprocess_ms.saturating_add(trim_ms);
                image_timing.total_ms = image_timing.total_ms.saturating_add(trim_ms);
                let render_to_image_ms = image_timing.total_ms;
                let (bytes, png_encode_ms) =
                    encode_rgba_png_profile_with_options(&trimmed, png_options);
                Ok((
                    bytes,
                    RenderPngTiming {
                        image_timing,
                        render_to_image_ms,
                        png_encode_ms,
                        png_write_ms: 0,
                        total_ms: render_to_image_ms.saturating_add(png_encode_ms),
                    },
                ))
            })?;
        let path = output_path.as_ref();
        let write_start = Instant::now();
        std::fs::write(path, bytes).map_err(|source| RustwxRenderError::WriteFile {
            path: path.display().to_string(),
            source,
        })?;
        let file_write_ms = write_start.elapsed().as_millis();
        let mut png_timing = png_timing;
        png_timing.png_write_ms = file_write_ms;
        let (georeference, georeference_absent_reason) =
            panel_georeference_for_save(request, &png_timing.image_timing);
        Ok(RenderSaveTiming {
            state_timing,
            png_timing,
            file_write_ms,
            total_ms: total_start.elapsed().as_millis(),
            georeference,
            georeference_absent_reason,
        })
    }
}

pub fn render_png(request: &MapRenderRequest) -> Result<Vec<u8>, RustwxRenderError> {
    RustRenderer.render_png(request)
}

pub fn render_image(request: &MapRenderRequest) -> Result<RgbaImage, RustwxRenderError> {
    RustRenderer.render_image(request)
}

pub fn render_image_with_style(
    request: &MapRenderRequest,
    plot_style: StaticPlotStyle,
) -> Result<RgbaImage, RustwxRenderError> {
    RustRenderer.render_image_with_style(request, plot_style)
}

pub fn save_png<P: AsRef<Path>>(
    request: &MapRenderRequest,
    output_path: P,
) -> Result<(), RustwxRenderError> {
    RustRenderer.save_png(request, output_path)
}

pub fn save_png_profile<P: AsRef<Path>>(
    request: &MapRenderRequest,
    output_path: P,
) -> Result<RenderSaveTiming, RustwxRenderError> {
    RustRenderer.save_png_profile(request, output_path)
}

pub fn save_png_profile_with_options<P: AsRef<Path>>(
    request: &MapRenderRequest,
    output_path: P,
    png_options: &PngWriteOptions,
) -> Result<RenderSaveTiming, RustwxRenderError> {
    RustRenderer.save_png_profile_with_options(request, output_path, png_options)
}

pub fn save_png_profile_with_options_and_style<P: AsRef<Path>>(
    request: &MapRenderRequest,
    output_path: P,
    png_options: &PngWriteOptions,
    plot_style: StaticPlotStyle,
) -> Result<RenderSaveTiming, RustwxRenderError> {
    RustRenderer.save_png_profile_with_options_and_style(
        request,
        output_path,
        png_options,
        plot_style,
    )
}

pub fn save_rgba_png_profile_with_options<P: AsRef<Path>>(
    image: &RgbaImage,
    output_path: P,
    png_options: &PngWriteOptions,
) -> Result<RenderSaveTiming, RustwxRenderError> {
    let total_start = Instant::now();
    let (bytes, png_encode_ms) = encode_rgba_png_profile_with_options(image, png_options);
    let path = output_path.as_ref();
    let write_start = Instant::now();
    std::fs::write(path, bytes).map_err(|source| RustwxRenderError::WriteFile {
        path: path.display().to_string(),
        source,
    })?;
    let file_write_ms = write_start.elapsed().as_millis();
    Ok(RenderSaveTiming {
        state_timing: RenderStateTiming::default(),
        png_timing: RenderPngTiming {
            image_timing: RenderImageTiming::default(),
            render_to_image_ms: 0,
            png_encode_ms,
            png_write_ms: file_write_ms,
            total_ms: png_encode_ms + file_write_ms,
        },
        file_write_ms,
        total_ms: total_start.elapsed().as_millis(),
        georeference: None,
        georeference_absent_reason: Some(
            "the PNG was saved from an already-composed RGBA canvas, not a single map \
             render; no one plot rectangle describes it"
                .to_string(),
        ),
    })
}

fn with_render_state<T>(
    request: &MapRenderRequest,
    render: impl FnOnce(&[f64], usize, usize, &RenderOpts) -> Result<T, RustwxRenderError>,
) -> Result<T, RustwxRenderError> {
    with_render_state_with_style(request, StaticPlotStyle::from_env(), render)
}

fn with_render_state_with_style<T>(
    request: &MapRenderRequest,
    plot_style: StaticPlotStyle,
    render: impl FnOnce(&[f64], usize, usize, &RenderOpts) -> Result<T, RustwxRenderError>,
) -> Result<T, RustwxRenderError> {
    with_render_state_profile_with_style(request, plot_style, |data, ny, nx, opts| {
        Ok((render(data, ny, nx, opts)?, RenderPngTiming::default()))
    })
    .map(|(result, _, _)| result)
}

fn with_render_state_profile_with_style<T>(
    request: &MapRenderRequest,
    plot_style: StaticPlotStyle,
    render: impl FnOnce(
        &[f64],
        usize,
        usize,
        &RenderOpts,
    ) -> Result<(T, RenderPngTiming), RustwxRenderError>,
) -> Result<(T, RenderStateTiming, RenderPngTiming), RustwxRenderError> {
    let total_start = Instant::now();
    let validate_start = Instant::now();
    validate_request(request)?;
    let validate_ms = validate_start.elapsed().as_millis();

    let shape = request.field.grid.shape;
    let overlay_only = request.is_overlay_only();
    let visual_mode = if overlay_only {
        ProductVisualMode::OverlayAnalysis
    } else {
        request.visual_mode
    };
    let presentation = RenderPresentation::for_mode_with_style(visual_mode, plot_style);
    let cmap = if overlay_only {
        blank_fill_colormap()
    } else {
        build_colormap(
            &request.scale,
            ColormapBuildOptions {
                render_density: plot_style.render_density(request.render_density),
                legend: request.legend,
            },
        )
    };
    let projected_domain = request.projected_domain.as_ref();
    let default_title = default_title(&request.field);

    RENDER_SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();

        let data_start = Instant::now();
        let data = if overlay_only {
            scratch.fill_f64_constant(shape.len(), OVERLAY_ONLY_FILL_VALUE)
        } else {
            scratch.fill_f64_from_f32(&request.field.values)
        };
        let data_buffer_ms = data_start.elapsed().as_millis();

        let projected_grid_start = Instant::now();
        let projected_grid = projected_domain.map(|domain| ProjectedGrid {
            x: scratch.fill_f64_from_f64(&domain.x),
            y: scratch.fill_f64_from_f64(&domain.y),
            ny: shape.ny,
            nx: shape.nx,
        });
        let projected_grid_ms = projected_grid_start.elapsed().as_millis();

        let inverse_projected_grid =
            request
                .inverse_raster_projection
                .as_ref()
                .and_then(|inverse| {
                    let projector = inverse
                        .projection
                        .build_projector(
                            inverse.reference_latitude_deg,
                            inverse.reference_longitude_deg,
                            &request.field.grid.lat_deg,
                            &request.field.grid.lon_deg,
                        )
                        .ok()?;
                    Some(InverseProjectedGrid {
                        projector,
                        clip_bounds: inverse.clip_bounds,
                        lat_deg: scratch.fill_f64_from_f32(&request.field.grid.lat_deg),
                        lon_deg: scratch.fill_f64_from_f32(&request.field.grid.lon_deg),
                    })
                });

        let rgba_grid = request.rgba_grid.as_ref().map(|field| {
            field
                .pixels
                .iter()
                .map(|pixel| Rgba {
                    r: pixel.r,
                    g: pixel.g,
                    b: pixel.b,
                    a: pixel.a,
                })
                .collect::<Vec<_>>()
        });

        let projected_lines_start = Instant::now();
        let mut projected_lines = Vec::with_capacity(request.projected_lines.len());
        for line in &request.projected_lines {
            projected_lines.push(ProjectedPolyline {
                points: scratch.fill_point_buffer(&line.points),
                color: line.color.into(),
                width: line.width,
                role: line.role,
            });
        }
        let projected_lines_ms = projected_lines_start.elapsed().as_millis();

        let projected_polygons_start = Instant::now();
        let mut projected_polygons = Vec::with_capacity(request.projected_polygons.len());
        for poly in &request.projected_polygons {
            let rings = poly
                .rings
                .iter()
                .map(|ring| scratch.fill_point_buffer(ring))
                .collect();
            projected_polygons.push(ProjectedPolygon {
                rings,
                color: poly.color.into(),
                role: poly.role,
            });
        }
        let projected_polygons_ms = projected_polygons_start.elapsed().as_millis();

        let mut projected_data_polygons = Vec::with_capacity(request.projected_data_polygons.len());
        for poly in &request.projected_data_polygons {
            let rings = poly
                .rings
                .iter()
                .map(|ring| scratch.fill_point_buffer(ring))
                .collect();
            projected_data_polygons.push(ProjectedPolygon {
                rings,
                color: poly.color.into(),
                role: poly.role,
            });
        }

        let mut projected_place_labels = Vec::with_capacity(request.projected_place_labels.len());
        for place_label in &request.projected_place_labels {
            projected_place_labels.push(ProjectedPlaceLabelOverlay {
                x: place_label.x,
                y: place_label.y,
                label: place_label.label.clone(),
                priority: place_label.priority,
                style: crate::overlay::ProjectedPlaceLabelStyle {
                    marker_radius_px: place_label.style.marker_radius_px,
                    marker_fill: place_label.style.marker_fill.into(),
                    marker_outline: place_label.style.marker_outline.into(),
                    marker_outline_width: place_label.style.marker_outline_width,
                    label_color: place_label.style.label_color.into(),
                    label_halo: place_label.style.label_halo.into(),
                    label_halo_width_px: place_label.style.label_halo_width_px,
                    label_scale: place_label.style.label_scale,
                    label_offset_x_px: place_label.style.label_offset_x_px,
                    label_offset_y_px: place_label.style.label_offset_y_px,
                    label_placement: place_label.style.label_placement,
                    label_bold: place_label.style.label_bold,
                },
            });
        }

        let projected_points = request
            .projected_points
            .iter()
            .map(|point| RenderProjectedPointOverlay {
                x: point.x,
                y: point.y,
                color: point.color.into(),
                radius_px: point.radius_px,
                width_px: point.width_px,
                shape: point.shape,
            })
            .collect::<Vec<_>>();

        let contour_start = Instant::now();
        let mut contours = Vec::with_capacity(request.contours.len());
        for layer in &request.contours {
            contours.push(ContourOverlay {
                data: scratch.fill_f64_from_f32(&layer.data),
                ny: shape.ny,
                nx: shape.nx,
                levels: scratch.fill_f64_from_f64(&layer.levels),
                color: presentation.contour_color(layer.color.into()),
                width: layer.width,
                labels: layer.labels,
                show_extrema: layer.show_extrema,
                pattern: layer.pattern,
                major_every: layer.major_every,
                major_width: layer.major_width,
            });
        }
        let contour_prep_ms = contour_start.elapsed().as_millis();

        let barb_start = Instant::now();
        let mut barbs = Vec::with_capacity(request.wind_barbs.len());
        for layer in &request.wind_barbs {
            barbs.push(BarbOverlay {
                u: layer.u.clone(),
                v: layer.v.clone(),
                ny: shape.ny,
                nx: shape.nx,
                stride_x: layer.stride_x,
                stride_y: layer.stride_y,
                spacing_px: layer.spacing_px,
                color: presentation.barb_color(layer.color.into()),
                halo_color: layer.halo_color.into(),
                halo_width: layer.halo_width,
                width: layer.width,
                length_px: layer.length_px,
            });
        }
        let mut streamlines = Vec::with_capacity(request.wind_streamlines.len());
        for layer in &request.wind_streamlines {
            streamlines.push(crate::overlay::StreamlineOverlay {
                u: scratch.fill_f64_from_f32(&layer.u),
                v: scratch.fill_f64_from_f32(&layer.v),
                ny: shape.ny,
                nx: shape.nx,
                stride_x: layer.stride_x,
                stride_y: layer.stride_y,
                color: presentation.barb_color(layer.color.into()),
                width: layer.width,
                max_steps: layer.max_steps,
                step_cells: layer.step_cells,
                min_speed: layer.min_speed,
            });
        }
        let barb_prep_ms = barb_start.elapsed().as_millis();

        let opts = RenderOpts {
            width: request.width,
            height: request.height,
            cmap,
            background: request.background.into(),
            colorbar: request.colorbar,
            title: request.title.clone().or(default_title),
            subtitle_left: request.subtitle_left.clone(),
            subtitle_center: request.subtitle_center.clone(),
            subtitle_right: request.subtitle_right.clone(),
            cbar_tick_step: request.cbar_tick_step,
            colorbar_mode: request.legend.mode,
            chrome_scale: request.chrome_scale,
            supersample_factor: plot_style.supersample_factor(request.supersample_factor),
            supersample_sharpen: plot_style.supersample_sharpen(request.supersample_sharpen),
            raster_sample_mode: request.raster_sample_mode,
            domain_frame: request.domain_frame,
            map_extent: projected_domain.map(|domain| MapExtent {
                x_min: domain.extent.x_min,
                x_max: domain.extent.x_max,
                y_min: domain.extent.y_min,
                y_max: domain.extent.y_max,
            }),
            projected_grid,
            inverse_projected_grid,
            rgba_grid,
            projected_polygons,
            projected_data_polygons,
            projected_place_labels,
            projected_points,
            projected_lines,
            contours,
            barbs,
            streamlines,
            presentation,
        };

        let state_timing = RenderStateTiming {
            validate_ms,
            data_buffer_ms,
            projected_grid_ms,
            projected_lines_ms,
            projected_polygons_ms,
            contour_prep_ms,
            barb_prep_ms,
            state_prep_ms: total_start.elapsed().as_millis(),
        };

        let result = render(&data, shape.ny, shape.nx, &opts);
        scratch.reclaim_render_opts(opts, data);
        result.map(|(value, png_timing)| (value, state_timing, png_timing))
    })
}

/// Build the exact [`LeveledColormap`] the rasterizer uses for `scale` under
/// `options` — the same function the PNG render path calls, exposed so
/// external viewers (e.g. the egui data viewer) can color values with
/// literally `cmap.map(value)` and stay bit-identical to the plot output.
///
/// To match a production render, pass the request's options filtered through
/// the active plot style:
/// `ColormapBuildOptions { render_density: StaticPlotStyle::from_env().render_density(request.render_density), legend: request.legend }`.
pub fn build_colormap(scale: &ColorScale, options: ColormapBuildOptions) -> LeveledColormap {
    let discrete = scale.resolved_discrete();

    let colors: Vec<Rgba> = discrete.colors.into_iter().map(Into::into).collect();
    LeveledColormap::from_palette_with_options(
        &colors,
        &discrete.levels,
        discrete.extend.into(),
        discrete.mask_below,
        options,
    )
}

/// The colorbar tick VALUES the PNG renderer would label for `cmap` with the
/// request's `cbar_tick_step` — the same `pick_ticks` over the same legend
/// levels the production colorbar uses. Label each value with
/// [`format_tick`] and position it at [`legend_tick_rel`] to reproduce the
/// production colorbar's numbers exactly.
pub fn colorbar_ticks(cmap: &LeveledColormap, cbar_tick_step: Option<f64>) -> Vec<f64> {
    crate::render::pick_ticks(cmap.legend_levels_for_display(), cbar_tick_step)
}

const OVERLAY_ONLY_FILL_VALUE: f64 = 0.5;

fn blank_fill_colormap() -> LeveledColormap {
    static BLANK_FILL_COLORMAP: OnceLock<LeveledColormap> = OnceLock::new();
    BLANK_FILL_COLORMAP
        .get_or_init(|| {
            LeveledColormap::from_palette(&[Rgba::TRANSPARENT], &[0.0, 1.0], Extend::Neither, None)
        })
        .clone()
}

fn default_title(field: &Field2D) -> Option<String> {
    match &field.product {
        ProductKey::Named(name) if !name.is_empty() => Some(name.clone()),
        _ => None,
    }
}

fn validate_request(request: &MapRenderRequest) -> Result<(), RustwxRenderError> {
    let expected = request.field.grid.shape.len();

    if let Some(rgba_grid) = &request.rgba_grid {
        if rgba_grid.grid.shape != request.field.grid.shape || rgba_grid.pixels.len() != expected {
            return Err(RustwxRenderError::LayerShapeMismatch {
                layer: "rgba_grid",
                expected,
                actual: rgba_grid.pixels.len(),
            });
        }
    }

    if let Some(domain) = &request.projected_domain {
        if request.field.grid.shape.nx < 2 || request.field.grid.shape.ny < 2 {
            return Err(RustwxRenderError::DegenerateProjectedGrid);
        }
        if domain.x.len() != domain.y.len() {
            return Err(RustwxRenderError::InvalidProjectedGrid);
        }
        if domain.x.len() != expected {
            return Err(RustwxRenderError::LayerShapeMismatch {
                layer: "projected_domain",
                expected,
                actual: domain.x.len(),
            });
        }
    }

    for layer in &request.contours {
        if layer.data.len() != expected {
            return Err(RustwxRenderError::LayerShapeMismatch {
                layer: "contour",
                expected,
                actual: layer.data.len(),
            });
        }
    }

    for layer in &request.wind_barbs {
        if layer.u.len() != expected {
            return Err(RustwxRenderError::LayerShapeMismatch {
                layer: "wind_barb_u",
                expected,
                actual: layer.u.len(),
            });
        }
        if layer.v.len() != expected {
            return Err(RustwxRenderError::LayerShapeMismatch {
                layer: "wind_barb_v",
                expected,
                actual: layer.v.len(),
            });
        }
    }

    for layer in &request.wind_streamlines {
        if layer.u.len() != expected {
            return Err(RustwxRenderError::LayerShapeMismatch {
                layer: "wind_streamline_u",
                expected,
                actual: layer.u.len(),
            });
        }
        if layer.v.len() != expected {
            return Err(RustwxRenderError::LayerShapeMismatch {
                layer: "wind_streamline_v",
                expected,
                actual: layer.v.len(),
            });
        }
    }

    Ok(())
}

impl From<Color> for Rgba {
    fn from(value: Color) -> Self {
        Self {
            r: value.r,
            g: value.g,
            b: value.b,
            a: value.a,
        }
    }
}

impl From<Rgba> for Color {
    fn from(value: Rgba) -> Self {
        Self {
            r: value.r,
            g: value.g,
            b: value.b,
            a: value.a,
        }
    }
}

impl From<ExtendMode> for Extend {
    fn from(value: ExtendMode) -> Self {
        match value {
            ExtendMode::Neither => Self::Neither,
            ExtendMode::Min => Self::Min,
            ExtendMode::Max => Self::Max,
            ExtendMode::Both => Self::Both,
        }
    }
}

pub fn draw_centered_text_line(img: &mut RgbaImage, text: &str, y: i32, color: Color, scale: u32) {
    text::draw_text_centered(img, text, y, color.into(), scale);
}

pub fn draw_centered_text_line_with_factor(
    img: &mut RgbaImage,
    text: &str,
    y: i32,
    color: Color,
    scale: u32,
    size_factor: f32,
) {
    let width = text::text_width_bold_with_factor(text, scale, size_factor) as i32;
    let x = ((img.width() as i32) - width) / 2;
    text::draw_text_bold_with_factor(img, text, x, y, color.into(), scale, size_factor);
}

pub fn draw_text_line(img: &mut RgbaImage, text: &str, x: i32, y: i32, color: Color, scale: u32) {
    text::draw_text(img, text, x, y, color.into(), scale);
}

pub fn draw_text_line_with_factor(
    img: &mut RgbaImage,
    text: &str,
    x: i32,
    y: i32,
    color: Color,
    scale: u32,
    size_factor: f32,
) {
    text::draw_text_with_factor(img, text, x, y, color.into(), scale, size_factor);
}

pub fn draw_right_text_line(
    img: &mut RgbaImage,
    text: &str,
    x_right: i32,
    y: i32,
    color: Color,
    scale: u32,
) {
    text::draw_text_right(img, text, x_right, y, color.into(), scale);
}

pub fn draw_right_text_line_with_factor(
    img: &mut RgbaImage,
    text: &str,
    x_right: i32,
    y: i32,
    color: Color,
    scale: u32,
    size_factor: f32,
) {
    let width = text::text_width_with_factor(text, scale, size_factor) as i32;
    text::draw_text_with_factor(
        img,
        text,
        x_right - width,
        y,
        color.into(),
        scale,
        size_factor,
    );
}

#[cfg(test)]
mod tests;
