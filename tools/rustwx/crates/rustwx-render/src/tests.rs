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
