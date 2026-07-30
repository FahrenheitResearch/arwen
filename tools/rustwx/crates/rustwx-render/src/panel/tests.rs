use super::*;
use crate::{
    ColorScale, ContourStyle, Field2D, GridShape, LatLonGrid, ProductKey, ProjectedDomain,
    ProjectedExtent, ProjectedLineOverlay, WindBarbStyle,
};
use image::ImageFormat;
use std::time::Instant;

fn solid_panel(width: u32, height: u32, rgba: [u8; 4]) -> RgbaImage {
    RgbaImage::from_pixel(width, height, Rgba(rgba))
}

fn sample_request(product: &str, width: u32, height: u32) -> MapRenderRequest {
    let shape = GridShape::new(3, 2).unwrap();
    let grid = LatLonGrid::new(
        shape,
        vec![35.0, 35.0, 35.0, 36.0, 36.0, 36.0],
        vec![-99.0, -98.0, -97.0, -99.0, -98.0, -97.0],
    )
    .unwrap();
    let field = Field2D::new(
        ProductKey::named(product),
        "J/kg",
        grid,
        vec![0.0, 250.0, 500.0, 750.0, 1000.0, 1250.0],
    )
    .unwrap();
    let mut request = MapRenderRequest::new(
        field,
        ColorScale::Weather(crate::weather::WeatherPreset::Cape),
    );
    request.width = width;
    request.height = height;
    request.colorbar = false;
    request
}

fn sample_projected_request(product: &str, width: u32, height: u32) -> MapRenderRequest {
    let mut request = sample_request(product, width, height);
    request.projected_domain = Some(ProjectedDomain {
        x: vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        y: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        extent: ProjectedExtent {
            x_min: 0.0,
            x_max: 2.0,
            y_min: 0.0,
            y_max: 1.0,
        },
    });
    request.projected_lines = vec![ProjectedLineOverlay {
        points: vec![(0.0, 0.0), (2.0, 1.0)],
        color: Color::BLACK,
        width: 2,
        role: crate::presentation::LineworkRole::Generic,
    }];
    request
}

fn sample_two_by_four_requests(width: u32, height: u32) -> Vec<MapRenderRequest> {
    vec![
        sample_projected_request("sbecape", width, height),
        sample_projected_request("mlecape", width, height),
        sample_projected_request("mucape", width, height),
        sample_projected_request("sbcin", width, height),
        sample_projected_request("mlcin", width, height),
        sample_projected_request("scp", width, height),
        sample_projected_request("stp", width, height),
        sample_projected_request("ehi", width, height),
    ]
}

fn render_panel_grid_legacy(
    layout: &PanelGridLayout,
    requests: &[MapRenderRequest],
) -> Result<RgbaImage, RustwxRenderError> {
    if requests.len() > layout.capacity() {
        return Err(RustwxRenderError::TooManyPanels {
            actual: requests.len(),
            capacity: layout.capacity(),
        });
    }

    let mut panels = Vec::with_capacity(requests.len());
    for (index, request) in requests.iter().enumerate() {
        validate_panel_size(layout, index, request.width, request.height)?;
        let png = crate::render_png(request)?;
        let panel = image::load_from_memory_with_format(&png, ImageFormat::Png)
            .map_err(|source| RustwxRenderError::DecodeRenderedPng { source })?
            .to_rgba8();
        panels.push(panel);
    }

    compose_panel_images(layout, &panels)
}

#[test]
fn compose_panel_images_places_row_major_panels_with_padding_and_gaps() {
    let layout = PanelGridLayout::two_by_two(4, 3)
        .unwrap()
        .with_gaps(1, 2)
        .with_padding(PanelPadding {
            top: 5,
            right: 4,
            bottom: 3,
            left: 2,
        })
        .with_background(Color::BLACK);

    let red = solid_panel(4, 3, [255, 0, 0, 255]);
    let green = solid_panel(4, 3, [0, 255, 0, 255]);
    let blue = solid_panel(4, 3, [0, 0, 255, 255]);
    let yellow = solid_panel(4, 3, [255, 255, 0, 255]);
    let canvas = compose_panel_images(&layout, &[red, green, blue, yellow]).unwrap();

    assert_eq!(canvas.width(), 15);
    assert_eq!(canvas.height(), 16);
    assert_eq!(canvas.get_pixel(0, 0).0, [0, 0, 0, 255]);
    assert_eq!(canvas.get_pixel(2, 5).0, [255, 0, 0, 255]);
    assert_eq!(canvas.get_pixel(7, 5).0, [0, 255, 0, 255]);
    assert_eq!(canvas.get_pixel(2, 10).0, [0, 0, 255, 255]);
    assert_eq!(canvas.get_pixel(7, 10).0, [255, 255, 0, 255]);
}

#[test]
fn compose_panel_images_rejects_size_mismatch() {
    let layout = PanelGridLayout::two_by_two(4, 3).unwrap();
    let panels = [
        solid_panel(4, 3, [255, 0, 0, 255]),
        solid_panel(5, 3, [0, 255, 0, 255]),
    ];
    let error = compose_panel_images(&layout, &panels).unwrap_err();

    assert!(matches!(
        error,
        RustwxRenderError::PanelSizeMismatch {
            index: 1,
            expected_width: 4,
            expected_height: 3,
            actual_width: 5,
            actual_height: 3,
        }
    ));
}

#[test]
fn render_panel_grid_leaves_unused_slots_as_background() {
    let layout = PanelGridLayout::two_by_two(140, 100)
        .unwrap()
        .with_background(Color::rgba(240, 240, 240, 255));
    let requests = [
        sample_request("sbecape", 140, 100),
        sample_request("mlecape", 140, 100),
    ];

    let canvas = render_panel_grid(&layout, &requests).unwrap();

    assert_eq!(canvas.width(), 280);
    assert_eq!(canvas.height(), 200);
    assert_eq!(canvas.get_pixel(210, 150).0, [240, 240, 240, 255]);
    let non_background = canvas
        .pixels()
        .filter(|px| px.0 != [240, 240, 240, 255])
        .count();
    assert!(
        non_background > 5000,
        "rendered panel grid should contain plot content"
    );
}

#[test]
fn render_panel_grid_renders_repeated_projected_domains() {
    let layout = PanelGridLayout::new(1, 2, 140, 100)
        .unwrap()
        .with_background(Color::rgba(245, 245, 245, 255));
    let requests = [
        sample_projected_request("sbecape", 140, 100),
        sample_projected_request("mlecape", 140, 100),
    ];

    let canvas = render_panel_grid(&layout, &requests).unwrap();

    assert_eq!(canvas.width(), 280);
    assert_eq!(canvas.height(), 100);
    let non_background = canvas
        .pixels()
        .filter(|px| px.0 != [245, 245, 245, 255])
        .count();
    assert!(
        non_background > 7000,
        "projected multi-panel render should contain plot and overlay content"
    );
}

#[test]
fn render_panel_grid_supports_mixed_filled_and_overlay_only_requests() {
    let layout = PanelGridLayout::new(1, 2, 160, 120).unwrap();
    let filled = sample_request("temperature", 160, 120);
    let height_field = sample_request("height", 160, 120).field;
    let contour_field = sample_request("height_contours", 160, 120).field;
    let u_field = sample_request("u10", 160, 120).field;
    let v_field = sample_request("v10", 160, 120).field;
    let mut overlay_only = crate::MapRenderRequest::contour_only(height_field)
        .with_contour_field(
            &contour_field,
            vec![250.0, 500.0, 750.0, 1000.0],
            ContourStyle {
                labels: true,
                ..Default::default()
            },
        )
        .unwrap()
        .with_wind_barbs(
            &u_field,
            &v_field,
            WindBarbStyle {
                stride_x: 1,
                stride_y: 1,
                length_px: 14.0,
                ..Default::default()
            },
        )
        .unwrap();
    overlay_only.width = 160;
    overlay_only.height = 120;

    let canvas = render_panel_grid(&layout, &[filled, overlay_only]).unwrap();
    assert_eq!(canvas.width(), 320);
    assert_eq!(canvas.height(), 120);
    let left_non_white = count_non_white_region(&canvas, 0, 0, 160, 120);
    let right_non_white = count_non_white_region(&canvas, 160, 0, 160, 120);
    assert!(left_non_white > 500);
    assert!(right_non_white > 500);
}

fn count_non_white_region(image: &RgbaImage, x0: u32, y0: u32, width: u32, height: u32) -> usize {
    let mut count = 0;
    for y in y0..y0 + height {
        for x in x0..x0 + width {
            if image.get_pixel(x, y).0 != [255, 255, 255, 255] {
                count += 1;
            }
        }
    }
    count
}

#[test]
fn render_panel_grid_matches_legacy_png_roundtrip_output() {
    let layout = PanelGridLayout::two_by_four(140, 100).unwrap();
    let requests = sample_two_by_four_requests(140, 100);

    let legacy = render_panel_grid_legacy(&layout, &requests).unwrap();
    let current = render_panel_grid(&layout, &requests).unwrap();

    assert_eq!(legacy.dimensions(), current.dimensions());
    assert_eq!(legacy.as_raw(), current.as_raw());
}

#[test]
#[ignore]
fn benchmark_render_panel_grid_vs_legacy_roundtrip() {
    let layout = PanelGridLayout::two_by_four(140, 100).unwrap();
    let requests = sample_two_by_four_requests(140, 100);
    let runs = 12u32;

    let legacy_probe = render_panel_grid_legacy(&layout, &requests).unwrap();
    let current_probe = render_panel_grid(&layout, &requests).unwrap();
    assert_eq!(legacy_probe.as_raw(), current_probe.as_raw());

    let mut legacy_total = 0.0f64;
    let mut current_total = 0.0f64;

    for _ in 0..runs {
        let started = Instant::now();
        let _ = render_panel_grid_legacy(&layout, &requests).unwrap();
        legacy_total += started.elapsed().as_secs_f64() * 1000.0;
    }

    for _ in 0..runs {
        let started = Instant::now();
        let _ = render_panel_grid(&layout, &requests).unwrap();
        current_total += started.elapsed().as_secs_f64() * 1000.0;
    }

    let legacy_mean = legacy_total / runs as f64;
    let current_mean = current_total / runs as f64;
    let delta_ms = current_mean - legacy_mean;
    let delta_pct = if legacy_mean.abs() > f64::EPSILON {
        (delta_ms / legacy_mean) * 100.0
    } else {
        0.0
    };

    println!(
        "{{\"legacy_mean_ms\":{legacy_mean:.3},\"current_mean_ms\":{current_mean:.3},\"delta_ms\":{delta_ms:.3},\"delta_pct\":{delta_pct:.2}}}"
    );
}
