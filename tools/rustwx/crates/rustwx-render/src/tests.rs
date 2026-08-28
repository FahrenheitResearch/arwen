use super::*;
use image::ImageFormat;

fn sample_field(product: &str) -> Field2D {
    let shape = GridShape::new(4, 3).unwrap();
    let lat = vec![35.0; shape.len()];
    let lon = vec![-97.0; shape.len()];
    let grid = LatLonGrid::new(shape, lat, lon).unwrap();
    let values = vec![
        0.0, 250.0, 750.0, 1500.0, 2000.0, 2400.0, 2600.0, 2800.0, 3000.0, 3200.0, 3400.0, 3600.0,
    ];
    Field2D::new(ProductKey::named(product), "J/kg", grid, values).unwrap()
}

#[test]
fn weather_product_mapping_covers_ecape_and_severe_aliases() {
    assert_eq!(
        WeatherProduct::from_product_name("sbecape"),
        Some(WeatherProduct::Sbecape)
    );
    assert_eq!(
        WeatherProduct::from_product_name("mlecin"),
        Some(WeatherProduct::Mlecin)
    );
    assert_eq!(
        WeatherProduct::from_product_name("ecape_scp"),
        Some(WeatherProduct::EcapeScpExperimental)
    );
    assert_eq!(
        WeatherProduct::from_product_name("sb_ecape_derived_cape_ratio"),
        Some(WeatherProduct::SbEcapeDerivedCapeRatio)
    );
    assert_eq!(
        WeatherProduct::from_product_name("mu_ecape_native_cape_ratio"),
        Some(WeatherProduct::MuEcapeNativeCapeRatio)
    );
    assert_eq!(
        WeatherProduct::from_product_name("ecape_ehi"),
        Some(WeatherProduct::EcapeEhi01kmExperimental)
    );
    assert_eq!(
        WeatherProduct::from_product_name("ecape_ehi_0_3km"),
        Some(WeatherProduct::EcapeEhi03kmExperimental)
    );
}

#[test]
fn render_png_emits_valid_nonempty_image() {
    let request = MapRenderRequest {
        field: sample_field("sbecape"),
        rgba_grid: None,
        product_metadata: None,
        width: 320,
        height: 240,
        scale: ColorScale::Weather(crate::weather::WeatherPreset::Cape),
        background: Color::WHITE,
        colorbar: true,
        title: Some("SBECAPE".into()),
        subtitle_left: Some("HRRR 2026-04-14 20Z F00".into()),
        subtitle_center: Some("rustwx-render".into()),
        subtitle_right: Some("rustwx-render".into()),
        cbar_tick_step: Some(500.0),
        render_density: RenderDensity::default(),
        legend: LegendControls::default(),
        chrome_scale: ChromeScale::default(),
        supersample_factor: 1,
        supersample_sharpen: true,
        visual_mode: ProductVisualMode::FilledMeteorology,
        raster_sample_mode: RasterSampleMode::default(),
        domain_frame: None,
        projected_domain: None,
        projected_polygons: Vec::new(),
        projected_data_polygons: Vec::new(),
        inverse_raster_projection: None,
        resolved_projection: None,
        geographic_bounds: None,
        projected_place_labels: Vec::new(),
        projected_points: Vec::new(),
        projected_lines: Vec::new(),
        contours: Vec::new(),
        wind_barbs: Vec::new(),
        wind_streamlines: Vec::new(),
        semantics: None,
    };

    let png = render_png(&request).unwrap();
    assert!(png.starts_with(&[137, 80, 78, 71, 13, 10, 26, 10]));

    let image = image::load_from_memory_with_format(&png, ImageFormat::Png)
        .unwrap()
        .to_rgba8();
    assert_eq!(image.width(), 320);
    assert_eq!(image.height(), 240);

    let non_white = image
        .pixels()
        .filter(|px| px.0 != [255, 255, 255, 255])
        .count();
    assert!(non_white > 1000, "image should contain rendered content");
}

#[test]
fn save_png_writes_file() {
    let request = MapRenderRequest::for_weather_product(sample_field("scp"), WeatherProduct::Scp);

    let path = std::env::temp_dir().join(format!("rustwx-render-{}.png", std::process::id()));
    save_png(&request, &path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert!(bytes.starts_with(&[137, 80, 78, 71, 13, 10, 26, 10]));

    let _ = std::fs::remove_file(path);
}

#[test]
fn render_image_emits_rgba_canvas_without_png_decode_in_callers() {
    let request = MapRenderRequest {
        field: sample_field("mucape"),
        rgba_grid: None,
        product_metadata: None,
        width: 320,
        height: 240,
        scale: ColorScale::Weather(crate::weather::WeatherPreset::Cape),
        background: Color::WHITE,
        colorbar: false,
        title: Some("MUCAPE".into()),
        subtitle_left: None,
        subtitle_center: None,
        subtitle_right: None,
        cbar_tick_step: Some(500.0),
        render_density: RenderDensity::default(),
        legend: LegendControls::default(),
        chrome_scale: ChromeScale::default(),
        supersample_factor: 1,
        supersample_sharpen: true,
        visual_mode: ProductVisualMode::FilledMeteorology,
        raster_sample_mode: RasterSampleMode::default(),
        domain_frame: None,
        projected_domain: None,
        projected_polygons: Vec::new(),
        projected_data_polygons: Vec::new(),
        inverse_raster_projection: None,
        resolved_projection: None,
        geographic_bounds: None,
        projected_place_labels: Vec::new(),
        projected_points: Vec::new(),
        projected_lines: Vec::new(),
        contours: Vec::new(),
        wind_barbs: Vec::new(),
        wind_streamlines: Vec::new(),
        semantics: None,
    };

    let image = render_image(&request).unwrap();
    assert_eq!(image.width(), 320);
    assert_eq!(image.height(), 240);

    let non_white = image
        .pixels()
        .filter(|px| px.0 != [255, 255, 255, 255])
        .count();
    assert!(non_white > 1000, "image should contain rendered content");
}

#[test]
fn with_render_state_carries_projected_place_labels_into_render_opts() {
    let mut request = MapRenderRequest::contour_only(sample_field("overlay"));
    request.projected_domain = Some(ProjectedDomain {
        x: vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
        y: vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        extent: ProjectedExtent {
            x_min: 0.0,
            x_max: 3.0,
            y_min: 0.0,
            y_max: 2.0,
        },
    });
    request.projected_place_labels.push(
        ProjectedPlaceLabel::new(1.5, 1.0)
            .with_label("Tulsa")
            .with_priority(ProjectedPlaceLabelPriority::Micro),
    );

    let carried = with_render_state(&request, |_data, _ny, _nx, opts| {
        Ok((
            opts.projected_place_labels.len(),
            opts.projected_place_labels[0].label.clone(),
            opts.projected_place_labels[0].style.marker_radius_px,
            opts.projected_place_labels[0].priority,
        ))
    })
    .unwrap();

    assert_eq!(carried.0, 1);
    assert_eq!(carried.1.as_deref(), Some("Tulsa"));
    assert_eq!(carried.2, 3);
    assert_eq!(carried.3, ProjectedPlaceLabelPriority::Micro);
}

#[test]
fn for_weather_product_sets_expected_titles_for_experimental_fields() {
    let request = MapRenderRequest::for_weather_product(
        sample_field("ecape_scp"),
        WeatherProduct::EcapeScpExperimental,
    );

    assert_eq!(request.title.as_deref(), Some("ECAPE SCP (EXP)"));
    assert_eq!(request.cbar_tick_step, Some(5.0));
    assert!(matches!(
        request.scale,
        ColorScale::Weather(WeatherPreset::Scp)
    ));
}

#[test]
fn derived_product_builder_renders_signed_field_with_builtin_scale() {
    let shape = GridShape::new(4, 3).unwrap();
    let lat = vec![35.0; shape.len()];
    let lon = vec![-97.0; shape.len()];
    let grid = LatLonGrid::new(shape, lat, lon).unwrap();
    let field = Field2D::new(
        ProductKey::named("temperature_advection_850mb"),
        "K/hr",
        grid,
        vec![
            -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0,
        ],
    )
    .unwrap();

    let request = MapRenderRequest::for_derived_product(
        field,
        DerivedProductStyle::TemperatureAdvection850mb,
    );
    let image = render_image(&request).unwrap();

    let non_white = image
        .pixels()
        .filter(|px| px.0 != [255, 255, 255, 255])
        .count();
    assert!(non_white > 1000, "derived render should contain content");
}

/// A whole-earth mesh with one point per 10 degrees, plus the request
/// options a caller would build for it.  Robinson is pinned explicitly so
/// the test cannot drift with projection inference; the measured failure
/// this whole feature answers was on a Robinson panel.
fn global_robinson_mesh() -> (Vec<f32>, Vec<f32>, ProjectedMapBuildOptions) {
    let mut lat = Vec::new();
    let mut lon = Vec::new();
    for row in 0..18 {
        for col in 0..36 {
            lat.push(-85.0 + row as f32 * 10.0);
            lon.push(-175.0 + col as f32 * 10.0);
        }
    }
    let options = ProjectedMapBuildOptions::from_bounds((-180.0, 180.0, -90.0, 90.0), 1.6)
        .with_projection(ProjectionSpec::Robinson {
            central_meridian_deg: 0.0,
        })
        .without_basemap();
    (lat, lon, options)
}

fn global_request_with_projected_domain() -> (MapRenderRequest, ProjectedMapBuildOptions) {
    let (lat, lon, options) = global_robinson_mesh();
    let projected = build_projected_map_with_options(&lat, &lon, &options).unwrap();
    let resolved = resolved_projection_for_options(&lat, &lon, &options.domain).unwrap();
    let shape = GridShape::new(36, 18).unwrap();
    let values: Vec<f32> = (0..shape.len()).map(|value| value as f32).collect();
    let grid = LatLonGrid::new(shape, lat, lon).unwrap();
    let field = Field2D::new(ProductKey::named("global"), "K", grid, values).unwrap();
    let mut request = MapRenderRequest::new(
        field,
        ColorScale::Weather(crate::weather::WeatherPreset::Cape),
    );
    request.width = 800;
    request.height = 600;
    request.projected_domain = Some(projected.domain());
    request.resolved_projection = Some(resolved);
    request.geographic_bounds = Some((-180.0, 180.0, -90.0, 90.0));
    (request, options)
}

/// The round trip that proves the published transform IS the drawn
/// transform: for a global Robinson panel, `PanelGeoReference::
/// lonlat_to_pixel` must agree -- to well under a pixel -- with what the
/// renderer's own projection seam (`project_geographic_points_with_options`
/// through the panel's extent and plot rectangle) produces for the same
/// points.  If the published projection resolved a different central
/// meridian, or the published extent were a different box, these numbers
/// would disagree by tens to hundreds of pixels, which is exactly the
/// wrong-ocean failure this feature retires.
#[test]
fn published_georeference_matches_the_drawn_projection_on_a_global_panel() {
    let (request, options) = global_request_with_projected_domain();
    let (lat, lon, _) = global_robinson_mesh();

    let path = std::env::temp_dir().join(format!(
        "rustwx-georef-global-{}.png",
        std::process::id()
    ));
    let timing = save_png_profile(&request, &path).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(
        timing.georeference_absent_reason, None,
        "a fully-specified request must publish"
    );
    let georeference = timing
        .georeference
        .expect("resolved projection + projected domain + bounds must publish");
    assert_eq!(georeference.image_width_px, 800);
    assert_eq!(georeference.image_height_px, 600);

    let image_timing = &timing.png_timing.image_timing;
    assert!(image_timing.plot_rect_describes_the_png);
    let extent = &request.projected_domain.as_ref().unwrap().extent;
    // The Southern Ocean point the placed-grid cascade got wrong, plus
    // spread-out controls.
    let points = [
        (-60.079, 139.499),
        (0.0, 0.0),
        (45.0, 90.0),
        (30.0, -100.0),
    ];
    let projected_points =
        project_geographic_points_with_options(&lat, &lon, &options, &points).unwrap();
    for ((point_lat, point_lon), (x, y)) in points.iter().zip(projected_points) {
        let (px, py) = georeference
            .lonlat_to_pixel(*point_lat, *point_lon)
            .expect("point inside the frame must land on a pixel");
        // The renderer's own answer for the same point: its projected
        // coordinates normalised through the panel extent onto the plot
        // rectangle.
        let rx = (x - extent.x_min) / (extent.x_max - extent.x_min);
        let ry = 1.0 - (y - extent.y_min) / (extent.y_max - extent.y_min);
        let expected_x =
            image_timing.map_x as f64 + rx * (image_timing.map_w.saturating_sub(1)) as f64;
        let expected_y =
            image_timing.map_y as f64 + ry * (image_timing.map_h.saturating_sub(1)) as f64;
        assert!(
            (px - expected_x).abs() < 0.05 && (py - expected_y).abs() < 0.05,
            "published transform disagrees with the drawn one at \
             ({point_lat},{point_lon}): published ({px},{py}), drawn ({expected_x},{expected_y})"
        );
    }
}

/// Centroid of the pixels within a small channel distance of `color`, in
/// FULL-image coordinates: where a drawn marker of that color actually
/// sits in the finished PNG.  This is the ground truth the pixel gate
/// compares against -- image content, not any rectangle the code under
/// test reported.
fn marker_centroid(image: &RgbaImage, color: Color) -> Option<(f64, f64)> {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut count = 0usize;
    for (x, y, pixel) in image.enumerate_pixels() {
        let distance = u32::from(pixel.0[0].abs_diff(color.r))
            + u32::from(pixel.0[1].abs_diff(color.g))
            + u32::from(pixel.0[2].abs_diff(color.b));
        if distance < 90 && pixel.0[3] > 200 {
            sum_x += x as f64;
            sum_y += y as f64;
            count += 1;
        }
    }
    (count > 0).then(|| (sum_x / count as f64, sum_y / count as f64))
}

/// The regional pixel gate, both directions.  A regional Lambert panel
/// takes the map-viewport crop -- the biggest of the three post-render
/// passes -- and its plot rectangle must FOLLOW the map through it:
///
/// * markers drawn at known lat/lons are located IN THE WRITTEN PNG by
///   color, and the published `lonlat_to_pixel` must land on them to
///   well under two pixels (marker centroid quantisation);
/// * the tester is then tested: the PRE-crop rectangle (the old
///   behaviour, reconstructed exactly from the reported offsets) must
///   FAIL the same comparison by more than a pixel.  If it did not, the
///   crop moved nothing and this test would be measuring nothing.
///
/// A published-and-wrong georeference is worse than a withheld one; this
/// is the gate that keeps the adjustment honest.
#[test]
fn published_georeference_survives_the_crop_on_a_regional_lambert_panel() {
    let mut lat = Vec::new();
    let mut lon = Vec::new();
    for row in 0..16 {
        for col in 0..26 {
            lat.push(30.0 + row as f32);
            lon.push(-110.0 + col as f32);
        }
    }
    let bounds = (-110.0, -85.0, 30.0, 45.0);
    let mut options = ProjectedMapBuildOptions::from_bounds(bounds, 1.6)
        .with_projection(ProjectionSpec::LambertConformal {
            standard_parallel_1_deg: 33.0,
            standard_parallel_2_deg: 45.0,
            central_meridian_deg: -96.0,
        })
        .without_basemap();
    options.domain.reference_latitude_deg = Some(39.0);
    let projected = build_projected_map_with_options(&lat, &lon, &options).unwrap();
    let resolved = resolved_projection_for_options(&lat, &lon, &options.domain).unwrap();
    assert!(
        matches!(resolved, ResolvedProjection::LambertConformal { .. }),
        "the regional gate must run on a Lambert panel, got {resolved:?}"
    );

    let shape = GridShape::new(26, 16).unwrap();
    let values: Vec<f32> = (0..shape.len()).map(|value| value as f32).collect();
    let grid = LatLonGrid::new(shape, lat.clone(), lon.clone()).unwrap();
    let field = Field2D::new(ProductKey::named("georef_pixel_probe"), "K", grid, values).unwrap();
    let mut request = MapRenderRequest::contour_only(field);
    request.width = 800;
    request.height = 600;
    request.projected_domain = Some(projected.domain());
    request.resolved_projection = Some(resolved);
    request.geographic_bounds = Some(bounds);
    request.domain_frame = Some(DomainFrame::map_viewport_default());

    // Known points spread across the frame, each marked in a color
    // nothing else in this panel uses (no fill, no basemap; linework and
    // text are black on paper).
    let points = [(33.0, -105.0), (42.0, -89.0), (36.5, -97.0)];
    let colors = [
        Color::rgba(255, 0, 255, 255),
        Color::rgba(0, 200, 0, 255),
        Color::rgba(255, 140, 0, 255),
    ];
    let projected_points =
        project_geographic_points_with_options(&lat, &lon, &options, &points).unwrap();
    for ((x, y), color) in projected_points.iter().zip(colors) {
        request.projected_points.push(ProjectedPointOverlay {
            x: *x,
            y: *y,
            color,
            radius_px: 4,
            width_px: 3,
            shape: ProjectedMarkerShape::Plus,
        });
    }

    let path = std::env::temp_dir().join(format!(
        "rustwx-georef-regional-{}.png",
        std::process::id()
    ));
    let timing = save_png_profile(&request, &path).unwrap();
    let final_image = image::load_from_memory_with_format(
        &std::fs::read(&path).unwrap(),
        ImageFormat::Png,
    )
    .unwrap()
    .to_rgba8();
    let _ = std::fs::remove_file(&path);

    let image_timing = &timing.png_timing.image_timing;
    assert!(
        image_timing.postprocess_offset_x != 0 || image_timing.postprocess_offset_y != 0,
        "the crop moved nothing, so this test is not measuring the crop -- \
         change the panel until it does"
    );
    let georeference = timing
        .georeference
        .expect("a cropped regional panel must still publish its transform");
    assert_eq!(
        (georeference.image_width_px, georeference.image_height_px),
        (final_image.width(), final_image.height()),
        "the published image size must be the written file's, not the request's"
    );

    // The OLD behaviour, reconstructed exactly: the pre-crop rectangle on
    // the pre-crop canvas.
    let stale = PanelGeoReference::new(
        request.width,
        request.height,
        PlotRect {
            x: (i64::from(georeference.plot_rect_px.x) - image_timing.postprocess_offset_x)
                as u32,
            y: (i64::from(georeference.plot_rect_px.y) - image_timing.postprocess_offset_y)
                as u32,
            width: georeference.plot_rect_px.width,
            height: georeference.plot_rect_px.height,
        },
        georeference.projection,
        georeference.extent.clone(),
        georeference.geographic_bounds,
    );

    let mut worst_published: f64 = 0.0;
    let mut worst_stale: f64 = 0.0;
    for (point, color) in points.iter().zip(colors) {
        let truth = marker_centroid(&final_image, color)
            .unwrap_or_else(|| panic!("marker {color:?} not found in the written PNG"));
        let (px, py) = georeference
            .lonlat_to_pixel(point.0, point.1)
            .expect("a point inside the frame must land on a pixel");
        worst_published =
            worst_published.max(((px - truth.0).powi(2) + (py - truth.1).powi(2)).sqrt());
        let (sx, sy) = stale
            .lonlat_to_pixel(point.0, point.1)
            .expect("the stale rectangle still places the point somewhere");
        worst_stale = worst_stale.max(((sx - truth.0).powi(2) + (sy - truth.1).powi(2)).sqrt());
    }
    assert!(
        worst_published < 2.0,
        "published transform misses the drawn markers by {worst_published} px"
    );
    assert!(
        worst_stale > 1.0 && worst_stale > worst_published,
        "the un-adjusted rectangle must measurably fail this comparison or the test \
         proves nothing: stale error {worst_stale} px, published error {worst_published} px"
    );
}

/// The residual refusal.  On every normal path the three passes now
/// report their offsets and the rectangle follows the map, so nothing
/// reaches the `plot_rect_describes_the_png == false` branch -- the
/// condition is constructed directly here (a pass whose adjusted
/// rectangle would fall outside the written image) to pin that it still
/// withholds rather than publishes.
#[test]
fn an_unreportable_map_move_still_withholds_the_georeference() {
    let (request, _) = global_request_with_projected_domain();
    let mut image_timing = RenderImageTiming {
        map_x: 18,
        map_y: 46,
        map_w: 700,
        map_h: 500,
        image_w: 800,
        image_h: 600,
        plot_rect_describes_the_png: false,
        ..RenderImageTiming::default()
    };

    let (georeference, reason) = panel_georeference_for_save(&request, &image_timing);
    assert!(georeference.is_none(), "a retired rectangle must not publish");
    let reason = reason.expect("a withheld georeference must say why");
    assert!(
        reason.contains("moved the map"),
        "the reason must name the concrete breakage: {reason}"
    );

    // The same timing with the rectangle intact publishes -- the refusal
    // above is the flag's doing, nothing else's.
    image_timing.plot_rect_describes_the_png = true;
    let (georeference, reason) = panel_georeference_for_save(&request, &image_timing);
    assert!(georeference.is_some() && reason.is_none());
}

#[test]
fn contour_only_map_with_height_contours_and_barbs_renders_visible_overlays() {
    let base = sample_field("height");
    let contours = sample_field("height_contours");
    let u = sample_field("u_wind");
    let mut v = sample_field("v_wind");
    v.values.iter_mut().for_each(|value| *value = 10.0);

    let request = MapRenderRequest::contour_only(base)
        .with_contour_field(
            &contours,
            vec![500.0, 1500.0, 2500.0, 3500.0],
            ContourStyle {
                labels: true,
                ..Default::default()
            },
        )
        .unwrap()
        .with_wind_barbs(
            &u,
            &v,
            WindBarbStyle {
                stride_x: 2,
                stride_y: 2,
                ..Default::default()
            },
        )
        .unwrap();

    let image = render_image(&request).unwrap();
    let non_white = image
        .pixels()
        .filter(|px| px.0 != [255, 255, 255, 255])
        .count();
    assert!(
        non_white > 1000,
        "overlay-only render should remain visible"
    );
}
