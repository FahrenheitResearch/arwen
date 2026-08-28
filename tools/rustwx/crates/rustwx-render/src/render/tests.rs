use super::*;
use crate::colormap::{ColormapBuildOptions, Extend, LevelDensity};
use crate::presentation::StaticPlotStyle;

fn sample_cmap() -> LeveledColormap {
    LeveledColormap::from_palette(
        &[Rgba::new(0, 0, 255), Rgba::new(255, 0, 0)],
        &[0.0, 1.0, 2.0, 3.0],
        Extend::Neither,
        None,
    )
}

fn sample_masked_cmap() -> LeveledColormap {
    LeveledColormap::from_palette(
        &[Rgba::new(0, 0, 255), Rgba::new(255, 0, 0)],
        &[10.0, 20.0, 30.0],
        Extend::Neither,
        Some(10.0),
    )
}

fn sample_projected_grid() -> ProjectedGrid {
    ProjectedGrid {
        x: vec![0.0, 1.0, 0.0, 1.0],
        y: vec![0.0, 0.0, 1.0, 1.0],
        ny: 2,
        nx: 2,
    }
}

fn sample_projected_opts() -> RenderOpts {
    RenderOpts {
        width: 240,
        height: 160,
        cmap: sample_cmap(),
        background: Rgba::WHITE,
        colorbar: false,
        title: Some("Projected".into()),
        subtitle_left: None,
        subtitle_center: None,
        subtitle_right: None,
        cbar_tick_step: None,
        colorbar_mode: crate::colormap::LegendMode::Stepped,
        chrome_scale: ChromeScale::default(),
        supersample_factor: 1,
        supersample_sharpen: true,
        raster_sample_mode: RasterSampleMode::default(),
        domain_frame: None,
        map_extent: Some(MapExtent {
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
        }),
        projected_grid: Some(sample_projected_grid()),
        inverse_projected_grid: None,
        rgba_grid: None,
        projected_polygons: Vec::new(),
        projected_data_polygons: Vec::new(),
        projected_place_labels: Vec::new(),
        projected_points: Vec::new(),
        projected_lines: Vec::new(),
        contours: Vec::new(),
        barbs: Vec::new(),
        streamlines: Vec::new(),
        presentation: RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology),
    }
}

#[test]
fn vertical_colorbar_follows_domain_frame_horizontally() {
    let presentation = RenderPresentation::for_mode_with_style(
        ProductVisualMode::FilledMeteorology,
        StaticPlotStyle::OperationalBudget30s,
    );
    let layout = compute_layout(2400, 1600, true, true, presentation, ChromeScale::default());
    let frame = DomainFrame::map_viewport_default();
    let rect = LocalRect {
        min_x: 0,
        max_x: layout.map_w.saturating_sub(160),
        min_y: 0,
        max_y: layout.map_h.saturating_sub(1),
    };

    let (x, y, w) = colorbar_anchor_rect(
        &layout,
        ColorbarOrientation::VerticalRight,
        Some(frame),
        Some(rect),
    );
    let frame_gap = (2u32.saturating_mul(layout.text_scale.max(1))).clamp(4, 8);

    assert_eq!(x, layout.map_x + rect.max_x + frame_gap);
    assert!(x < layout.cbar_x);
    assert_eq!(y, layout.cbar_y);
    assert_eq!(w, layout.cbar_w);
}

fn sample_place_label() -> ProjectedPlaceLabelOverlay {
    ProjectedPlaceLabelOverlay {
        x: 0.52,
        y: 0.48,
        label: Some("Sacramento".into()),
        priority: ProjectedPlaceLabelPriority::Primary,
        style: crate::overlay::ProjectedPlaceLabelStyle {
            marker_radius_px: 4,
            marker_fill: Rgba::with_alpha(255, 255, 255, 235),
            marker_outline: Rgba::with_alpha(24, 28, 34, 240),
            marker_outline_width: 1,
            label_color: Rgba::BLACK,
            label_halo: Rgba::with_alpha(255, 255, 255, 235),
            label_halo_width_px: 2,
            label_scale: 1,
            label_offset_x_px: 6,
            label_offset_y_px: -2,
            label_placement: ProjectedLabelPlacement::AboveRight,
            label_bold: true,
        },
    }
}

fn contour_test_layout() -> Layout {
    Layout {
        map_x: 0,
        map_y: 0,
        map_w: 64,
        map_h: 64,
        cbar_x: 0,
        cbar_y: 0,
        cbar_w: 0,
        cbar_h: 0,
        title_y: 0,
        subtitle_y: 0,
        text_scale: 1,
        label_gap: 14,
    }
}

fn blank_test_image() -> RgbaImage {
    RgbaImage::from_pixel(80, 80, Rgba::WHITE.to_image_rgba())
}

fn non_white_bounds(img: &RgbaImage) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = u32::MAX;
    let mut max_x = 0u32;
    let mut min_y = u32::MAX;
    let mut max_y = 0u32;
    let mut found = false;

    for (x, y, pixel) in img.enumerate_pixels() {
        if pixel.0 == [255, 255, 255, 255] {
            continue;
        }
        found = true;
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    found.then_some((min_x, max_x, min_y, max_y))
}

fn sample_domain_frame(outline_color: crate::request::Color) -> DomainFrame {
    DomainFrame {
        inset_px: 5,
        outline_color,
        outline_width: 2,
        clear_outside: true,
        legend_follows_frame: true,
        chrome_follows_frame: true,
        source: crate::request::DomainFrameSource::ProjectedGrid,
    }
}

#[test]
fn projected_pixels_keep_nearby_offscreen_points_for_clipping() {
    let layout = contour_test_layout();
    let grid = ProjectedGrid {
        x: vec![-0.05, 1.05, -0.05, 1.05],
        y: vec![0.0, 0.0, 1.0, 1.0],
        ny: 2,
        nx: 2,
    };
    let extent = MapExtent {
        x_min: 0.0,
        x_max: 1.0,
        y_min: 0.0,
        y_max: 1.0,
    };

    let pixels = projected_grid_to_pixels(&grid, &extent, &layout);

    assert_eq!(pixels.len(), 4);
    assert!(pixels[0].is_some_and(|(x, _)| x < 0.0));
    assert!(pixels[1].is_some_and(|(x, _)| x > layout.map_w as f32 - 1.0));
}

fn visit_rs_files(
    root: &std::path::Path,
    visitor: &mut impl FnMut(&std::path::Path),
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            visit_rs_files(&path, visitor)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            visitor(&path);
        }
    }
    Ok(())
}

#[test]
fn supersample_scaling_expands_overlay_dimensions() {
    let mut opts = sample_projected_opts();
    opts.projected_lines = vec![ProjectedPolyline {
        points: vec![(0.0, 0.0), (1.0, 1.0)],
        color: Rgba::BLACK,
        width: 2,
        role: crate::presentation::LineworkRole::Generic,
    }];
    opts.projected_points = vec![ProjectedPointOverlay {
        x: 0.50,
        y: 0.50,
        color: Rgba::new(255, 80, 40),
        radius_px: 5,
        width_px: 2,
        shape: ProjectedMarkerShape::Plus,
    }];
    opts.contours = vec![ContourOverlay {
        data: vec![500.0, 504.0, 508.0, 512.0],
        ny: 2,
        nx: 2,
        levels: vec![504.0],
        color: Rgba::BLACK,
        width: 1,
        labels: false,
        show_extrema: false,
        pattern: crate::request::ContourLinePattern::Solid,
        major_every: Some(1),
        major_width: Some(3),
    }];
    opts.barbs = vec![BarbOverlay {
        u: vec![10.0, 10.0, 10.0, 10.0],
        v: vec![0.0, 0.0, 0.0, 0.0],
        ny: 2,
        nx: 2,
        stride_x: 1,
        stride_y: 1,
        spacing_px: 24.0,
        color: Rgba::BLACK,
        halo_color: Rgba::WHITE,
        halo_width: 2,
        width: 1,
        length_px: 18.0,
    }];
    opts.streamlines = vec![StreamlineOverlay {
        u: vec![10.0, 10.0, 10.0, 10.0],
        v: vec![0.0, 0.0, 0.0, 0.0],
        ny: 2,
        nx: 2,
        stride_x: 1,
        stride_y: 1,
        color: Rgba::with_alpha(0, 0, 0, 120),
        width: 1,
        max_steps: 4,
        step_cells: 0.5,
        min_speed: 2.5,
    }];
    opts.projected_place_labels = vec![sample_place_label()];
    opts.domain_frame = Some(sample_domain_frame(crate::request::Color::BLACK));

    let scaled = scale_render_opts_for_supersample(&opts, 2);
    assert_eq!(scaled.width, opts.width * 2);
    assert_eq!(scaled.height, opts.height * 2);
    assert_eq!(scaled.projected_lines[0].width, 4);
    assert_eq!(scaled.projected_points[0].radius_px, 10);
    assert_eq!(scaled.projected_points[0].width_px, 4);
    assert_eq!(scaled.projected_place_labels[0].style.marker_radius_px, 8);
    assert_eq!(scaled.projected_place_labels[0].style.label_scale, 2);
    assert_eq!(scaled.projected_place_labels[0].style.label_offset_x_px, 12);
    assert_eq!(scaled.contours[0].width, 2);
    assert_eq!(scaled.contours[0].major_width, Some(6));
    assert_eq!(scaled.barbs[0].width, 2);
    assert_eq!(scaled.barbs[0].halo_width, 4);
    assert_eq!(scaled.barbs[0].spacing_px, 48.0);
    assert_eq!(scaled.barbs[0].length_px, 36.0);
    assert_eq!(scaled.streamlines[0].width, 2);
    assert_eq!(scaled.domain_frame.unwrap().outline_width, 4);
    assert_eq!(scaled.supersample_factor, 1);
    assert_eq!(scaled.supersample_sharpen, opts.supersample_sharpen);
}

#[test]
fn wind_streamlines_draw_visible_flow_lines() {
    let mut image = blank_test_image();
    let layout = contour_test_layout();
    let overlay = StreamlineOverlay {
        u: vec![10.0; 64],
        v: vec![0.0; 64],
        ny: 8,
        nx: 8,
        stride_x: 2,
        stride_y: 2,
        color: Rgba::BLACK,
        width: 1,
        max_steps: 8,
        step_cells: 0.5,
        min_speed: 2.5,
    };

    draw_streamlines(&mut image, &layout, &overlay, None, None);

    let bounds = non_white_bounds(&image).expect("streamlines should draw");
    assert!(bounds.1 > bounds.0, "flow lines should span horizontally");
}

#[test]
fn render_to_image_supersample_preserves_requested_dimensions() {
    let mut opts = sample_projected_opts();
    opts.supersample_factor = 2;
    let data = vec![10.0, 20.0, 30.0, 25.0];
    let (image, timing) = render_to_image_profile(&data, 2, 2, &opts);
    assert_eq!(image.width(), opts.width);
    assert_eq!(image.height(), opts.height);
    assert!(timing.postprocess_ms <= timing.total_ms);
}

#[test]
fn render_to_image_supersample_can_skip_sharpen_pass() {
    let mut opts = sample_projected_opts();
    opts.supersample_factor = 2;
    opts.supersample_sharpen = false;
    let data = vec![10.0, 20.0, 30.0, 25.0];

    let (image, timing) = render_to_image_profile(&data, 2, 2, &opts);

    assert_eq!(image.width(), opts.width);
    assert_eq!(image.height(), opts.height);
    assert!(timing.downsample_ms <= timing.total_ms);
}

#[test]
fn projected_place_labels_render_visible_marker_and_text() {
    let mut opts = sample_projected_opts();
    opts.projected_place_labels = vec![sample_place_label()];
    let data = vec![0.5, 1.0, 1.5, 2.0];

    let image = render_to_image(&data, 2, 2, &opts);
    let dark_pixels = image
        .pixels()
        .filter(|pixel| pixel.0[0] < 80 && pixel.0[1] < 80 && pixel.0[2] < 80)
        .count();
    let bright_pixels = image
        .pixels()
        .filter(|pixel| pixel.0[0] > 220 && pixel.0[1] > 220 && pixel.0[2] > 220)
        .count();

    assert!(
        dark_pixels > 50,
        "label text and marker outline should be visible"
    );
    assert!(
        bright_pixels > 200,
        "marker fill and halo should be visible"
    );
}

#[test]
fn projected_points_render_visible_marker() {
    let mut opts = sample_projected_opts();
    opts.projected_points = vec![ProjectedPointOverlay {
        x: 0.50,
        y: 0.50,
        color: Rgba::new(255, 50, 20),
        radius_px: 8,
        width_px: 2,
        shape: ProjectedMarkerShape::Plus,
    }];
    let data = vec![0.5, 1.0, 1.5, 2.0];

    let image = render_to_image(&data, 2, 2, &opts);
    let red_pixels = image
        .pixels()
        .filter(|pixel| pixel.0[0] > 200 && pixel.0[1] < 100 && pixel.0[2] < 100)
        .count();

    assert!(red_pixels > 15, "projected point marker should be visible");
}

#[test]
fn projected_place_labels_clamp_text_inside_requested_clip_rect() {
    let presentation = RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology);
    let layout = compute_layout(240, 160, false, false, presentation, ChromeScale::default());
    let extent = MapExtent {
        x_min: 0.0,
        x_max: 1.0,
        y_min: 0.0,
        y_max: 1.0,
    };
    let clip_rect = LocalRect {
        min_x: 20,
        max_x: 80,
        min_y: 20,
        max_y: 60,
    };
    let local_x = clip_rect.max_x.saturating_sub(2) as f64;
    let local_y = clip_rect.max_y.saturating_sub(2) as f64;
    let mut style = sample_place_label().style;
    style.marker_radius_px = 0;
    style.marker_outline_width = 0;
    style.label_halo = Rgba::TRANSPARENT;
    style.label_halo_width_px = 0;
    style.label_offset_x_px = 28;
    style.label_offset_y_px = 14;
    style.label_placement = ProjectedLabelPlacement::BelowRight;
    let label = ProjectedPlaceLabelOverlay {
        x: local_x / layout.map_w.saturating_sub(1) as f64,
        y: 1.0 - (local_y / layout.map_h.saturating_sub(1) as f64),
        label: Some("Sacramento Valley".into()),
        priority: ProjectedPlaceLabelPriority::Primary,
        style,
    };
    let mut img = RgbaImage::from_pixel(240, 160, Rgba::WHITE.to_image_rgba());

    draw_projected_place_labels(&mut img, &layout, &extent, &[label], None, Some(clip_rect));

    let (min_x, max_x, min_y, max_y) =
        non_white_bounds(&img).expect("clipped place label should still render");
    assert!(min_x >= layout.map_x + clip_rect.min_x);
    assert!(max_x <= layout.map_x + clip_rect.max_x);
    assert!(min_y >= layout.map_y + clip_rect.min_y);
    assert!(max_y <= layout.map_y + clip_rect.max_y);
}

#[test]
fn projected_place_labels_skip_marker_and_text_outside_requested_clip_mask() {
    let presentation = RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology);
    let layout = compute_layout(240, 160, false, false, presentation, ChromeScale::default());
    let extent = MapExtent {
        x_min: 0.0,
        x_max: 1.0,
        y_min: 0.0,
        y_max: 1.0,
    };
    let clip_mask = RgbaImage::from_pixel(
        layout.map_w,
        layout.map_h,
        Rgba::TRANSPARENT.to_image_rgba(),
    );
    let mut img = RgbaImage::from_pixel(240, 160, Rgba::WHITE.to_image_rgba());

    draw_projected_place_labels(
        &mut img,
        &layout,
        &extent,
        &[sample_place_label()],
        Some(&clip_mask),
        None,
    );

    assert!(
        non_white_bounds(&img).is_none(),
        "place labels whose marker falls outside the clip mask should not render"
    );
}

#[test]
fn projected_place_label_priorities_reduce_auxiliary_and_micro_visual_weight() {
    let primary = place_label_render_adjustments(ProjectedPlaceLabelPriority::Primary);
    let auxiliary = place_label_render_adjustments(ProjectedPlaceLabelPriority::Auxiliary);
    let micro = place_label_render_adjustments(ProjectedPlaceLabelPriority::Micro);

    assert_eq!(primary.text_size_factor, 1.0);
    assert_eq!(primary.marker_scale_factor, 1.0);
    assert!(auxiliary.text_size_factor < primary.text_size_factor);
    assert!(auxiliary.text_alpha_factor < primary.text_alpha_factor);
    assert!(micro.text_size_factor < auxiliary.text_size_factor);
    assert!(micro.text_alpha_factor < auxiliary.text_alpha_factor);
    assert!(micro.marker_scale_factor < auxiliary.marker_scale_factor);
    assert!(micro.halo_width_factor < auxiliary.halo_width_factor);
}

fn slanted_projected_fixture() -> (Layout, ProjectedGrid, Arc<[Option<(f64, f64)>]>, LocalRect) {
    let layout = compute_layout(
        320,
        240,
        true,
        true,
        RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology),
        ChromeScale::default(),
    );
    let nx = 14usize;
    let ny = 10usize;
    let grid = ProjectedGrid {
        x: vec![0.0; nx * ny],
        y: vec![0.0; nx * ny],
        ny,
        nx,
    };
    let mut pixel_points = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            pixel_points.push(Some((
                28.0 + i as f64 * 12.0 + j as f64 * 0.5,
                10.0 + j as f64 * 8.0,
            )));
        }
    }
    let pixel_points: Arc<[Option<(f64, f64)>]> = pixel_points.into();
    let rect = compute_domain_frame_rect(
        sample_domain_frame(crate::request::Color::BLACK),
        layout.map_w,
        layout.map_h,
    )
    .expect("test layout should produce a frame rect");
    (layout, grid, pixel_points, rect)
}

#[test]
fn bucketed_contours_match_legacy_for_sorted_levels() {
    let layout = contour_test_layout();
    let overlay = ContourOverlay {
        data: vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        ny: 3,
        nx: 3,
        levels: vec![1.5, 2.5],
        color: Rgba::BLACK,
        width: 1,
        labels: false,
        show_extrema: false,
        pattern: crate::request::ContourLinePattern::Solid,
        major_every: None,
        major_width: None,
    };

    let mut legacy = blank_test_image();
    let mut bucketed = blank_test_image();
    let mut legacy_labels = ContourLabelPlacer::default();
    let mut bucketed_labels = ContourLabelPlacer::default();
    draw_contours_legacy(
        &mut legacy,
        &layout,
        &overlay,
        None,
        None,
        &mut legacy_labels,
        1,
    );
    draw_contours_bucketed(
        &mut bucketed,
        &layout,
        &overlay,
        None,
        None,
        &mut bucketed_labels,
        1,
    );

    assert_eq!(legacy, bucketed);
}

#[test]
fn bucketed_contours_match_legacy_with_nan_corner() {
    let layout = contour_test_layout();
    let overlay = ContourOverlay {
        data: vec![0.0, 1.0, f64::NAN, 3.0],
        ny: 2,
        nx: 2,
        levels: vec![0.5, 1.5, 2.5],
        color: Rgba::BLACK,
        width: 1,
        labels: false,
        show_extrema: false,
        pattern: crate::request::ContourLinePattern::Solid,
        major_every: None,
        major_width: None,
    };

    let mut legacy = blank_test_image();
    let mut bucketed = blank_test_image();
    let mut legacy_labels = ContourLabelPlacer::default();
    let mut bucketed_labels = ContourLabelPlacer::default();
    draw_contours_legacy(
        &mut legacy,
        &layout,
        &overlay,
        None,
        None,
        &mut legacy_labels,
        1,
    );
    draw_contours_bucketed(
        &mut bucketed,
        &layout,
        &overlay,
        None,
        None,
        &mut bucketed_labels,
        1,
    );

    assert_eq!(legacy, bucketed);
}

#[test]
fn contour_label_placer_rejects_overlapping_labels() {
    let mut placer = ContourLabelPlacer::default();
    assert!(placer.can_place(LabelRect {
        min_x: 20,
        max_x: 70,
        min_y: 20,
        max_y: 34,
    }));
    assert!(!placer.can_place(LabelRect {
        min_x: 68,
        max_x: 110,
        min_y: 21,
        max_y: 35,
    }));
    assert!(placer.can_place(LabelRect {
        min_x: 120,
        max_x: 160,
        min_y: 21,
        max_y: 35,
    }));
}

#[test]
fn contour_label_state_allows_repeated_spaced_labels_per_level() {
    let mut layout = contour_test_layout();
    layout.map_w = 1500;
    layout.map_h = 850;
    let mut state = ContourLevelLabelState::new(true, &layout);

    assert!(state.max_labels > 1);
    assert!(state.can_try_at((100.0, 100.0)));
    state.record((100.0, 100.0));
    assert!(!state.can_try_at((130.0, 120.0)));
    assert!(state.can_try_at((420.0, 100.0)));

    while state.centers.len() < state.max_labels {
        let x = 100.0 + state.centers.len() as f64 * 260.0;
        assert!(state.can_try_at((x, 650.0)));
        state.record((x, 650.0));
    }
    assert!(!state.can_try_at((5000.0, 5000.0)));
}

#[test]
fn contour_labels_only_use_major_levels_when_configured() {
    let overlay = ContourOverlay {
        data: Vec::new(),
        ny: 0,
        nx: 0,
        levels: vec![540.0, 546.0, 552.0, 558.0],
        color: Rgba::BLACK,
        width: 1,
        labels: true,
        show_extrema: false,
        pattern: crate::request::ContourLinePattern::Solid,
        major_every: Some(2),
        major_width: Some(2),
    };

    assert!(contour_level_gets_label(&overlay, 0));
    assert!(!contour_level_gets_label(&overlay, 1));
    assert!(contour_level_gets_label(&overlay, 2));
    assert!(!contour_level_gets_label(&overlay, 3));
}

#[test]
fn contour_stroke_supports_major_width_and_dashes() {
    let overlay = ContourOverlay {
        data: Vec::new(),
        ny: 0,
        nx: 0,
        levels: vec![1000.0, 1002.0, 1004.0],
        color: Rgba::BLACK,
        width: 1,
        labels: false,
        show_extrema: false,
        pattern: crate::request::ContourLinePattern::Solid,
        major_every: Some(2),
        major_width: Some(3),
    };
    assert_eq!(contour_level_width(&overlay, 0), 3);
    assert_eq!(contour_level_width(&overlay, 1), 1);
    assert_eq!(contour_level_width(&overlay, 2), 3);

    let mut solid = blank_test_image();
    let mut dashed = blank_test_image();
    draw_contour_stroke(
        &mut solid,
        5.0,
        40.0,
        75.0,
        40.0,
        Rgba::BLACK,
        1,
        crate::request::ContourLinePattern::Solid,
    );
    draw_contour_stroke(
        &mut dashed,
        5.0,
        40.0,
        75.0,
        40.0,
        Rgba::BLACK,
        1,
        crate::request::ContourLinePattern::Dashed,
    );
    let solid_pixels = solid
        .pixels()
        .filter(|pixel| pixel.0 != [255, 255, 255, 255])
        .count();
    let dashed_pixels = dashed
        .pixels()
        .filter(|pixel| pixel.0 != [255, 255, 255, 255])
        .count();

    assert!(dashed_pixels > 0);
    assert!(dashed_pixels < solid_pixels);
}

#[test]
fn domain_frame_uses_viewport_when_fill_is_fully_masked() {
    let mut opts = sample_projected_opts();
    opts.cmap = sample_masked_cmap();
    opts.title = None;
    opts.domain_frame = Some(sample_domain_frame(crate::request::Color::rgba(
        250, 10, 10, 255,
    )));

    let data = [0.0f64; 4];
    let (image, timing) = render_to_image_profile(&data, 2, 2, &opts);
    let outline_pixels = image
        .pixels()
        .filter(|px| px.0[0] > 180 && px.0[1] < 120 && px.0[2] < 120)
        .count();

    assert!(
        outline_pixels > 0,
        "domain frame should still render when fill alpha is empty"
    );
    assert_eq!(
        timing.domain_clip_rect,
        Some([
            5,
            timing.map_w.saturating_sub(7),
            5,
            timing.map_h.saturating_sub(7)
        ]),
        "domain frame should follow the map viewport, not the data coverage"
    );
}

#[test]
fn domain_frame_clears_map_outside_rect() {
    let (layout, _, _, rect) = slanted_projected_fixture();
    let presentation = RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology);
    let mut img = RgbaImage::from_pixel(320, 240, presentation.canvas_background.to_image_rgba());

    for py in layout.map_y..layout.map_y + layout.map_h {
        for px in layout.map_x..layout.map_x + layout.map_w {
            img.put_pixel(px, py, presentation.map_background.to_image_rgba());
        }
    }

    let outside_x = layout.map_x + rect.min_x.saturating_sub(1);
    let outside_y = layout.map_y + rect.min_y;
    let inside_x = layout.map_x + rect.min_x + 1;
    let inside_y = layout.map_y + rect.min_y + 1;
    img.put_pixel(outside_x, outside_y, Rgba::BLACK.to_image_rgba());

    clear_map_outside_local_rect(&mut img, &layout, rect, presentation.canvas_background);

    assert_eq!(
        img.get_pixel(outside_x, outside_y).0,
        presentation.canvas_background.to_image_rgba().0
    );
    assert_eq!(
        img.get_pixel(inside_x, inside_y).0,
        presentation.map_background.to_image_rgba().0
    );
}

#[test]
fn domain_frame_keeps_colorbar_in_layout_when_frame_matches_viewport() {
    let (layout, _, _, rect) = slanted_projected_fixture();
    let frame = sample_domain_frame(crate::request::Color::BLACK);

    let (_, cbar_y, _) = colorbar_anchor_rect(
        &layout,
        ColorbarOrientation::HorizontalBottom,
        Some(frame),
        Some(rect),
    );

    assert_eq!(cbar_y, layout.cbar_y);
}

#[test]
fn domain_frame_layout_reserves_space_for_legend_labels() {
    let layout = compute_effective_layout(
        1400,
        1100,
        true,
        true,
        RenderPresentation::for_mode(ProductVisualMode::OverlayAnalysis),
        ChromeScale::Fixed(1.0),
        true,
    );

    let label_top = layout.cbar_y.saturating_sub(layout.label_gap);
    let map_bottom = layout.map_y.saturating_add(layout.map_h).saturating_sub(1);
    assert!(layout.label_gap > text::regular_line_height(layout.text_scale));
    assert!(label_top > map_bottom);
}

#[test]
fn domain_frame_text_anchors_to_rect() {
    let (layout, _, _, rect) = slanted_projected_fixture();
    let frame = sample_domain_frame(crate::request::Color::BLACK);

    let (left, right, center) = chrome_anchor_bounds(&layout, Some(frame), Some(rect));

    assert_eq!(left, layout.map_x + rect.min_x);
    assert_eq!(right, layout.map_x + rect.max_x);
    assert_eq!(center, left + right.saturating_sub(left) / 2);
    assert_ne!(left, layout.map_x);
    assert_ne!(right, layout.map_x + layout.map_w);
}

#[test]
fn domain_frame_text_rows_anchor_just_above_rect() {
    let (layout, _, _, rect) = slanted_projected_fixture();
    let frame = sample_domain_frame(crate::request::Color::BLACK);

    let (title_y, subtitle_y) = chrome_anchor_rows(&layout, Some(frame), Some(rect));
    let frame_top = layout.map_y + rect.min_y;
    let max_gap = text::bold_line_height(layout.text_scale)
        .saturating_add(text::regular_line_height(layout.text_scale))
        .saturating_add(8u32.saturating_mul(layout.text_scale.max(1)));

    assert!(title_y <= subtitle_y);
    assert!(subtitle_y < frame_top);
    assert!(title_y < frame_top);
    assert!(frame_top.saturating_sub(title_y) <= max_gap);
}

#[test]
fn chrome_metadata_uses_space_left_by_short_title() {
    let metadata = "Init 05/04 11Z | F008 | Valid 05/04 19Z | HRRR | source: nomads";

    let (title, fitted_metadata) =
        fit_chrome_title_metadata(Some("2m AGL Temperature"), Some(metadata), 940, 14, 1);

    assert_eq!(title.as_deref(), Some("2m AGL Temperature"));
    assert_eq!(fitted_metadata.as_deref(), Some(metadata));
}

#[test]
fn projected_alpha_mask_clears_linework_outside_mask() {
    let layout = Layout {
        map_x: 1,
        map_y: 1,
        map_w: 4,
        map_h: 4,
        cbar_x: 0,
        cbar_y: 0,
        cbar_w: 0,
        cbar_h: 0,
        title_y: 0,
        subtitle_y: 0,
        text_scale: 1,
        label_gap: 1,
    };
    let bg = Rgba::new(244, 246, 248);
    let mut img = RgbaImage::from_pixel(6, 6, Rgba::BLACK.to_image_rgba());
    let mut mask = RgbaImage::new(4, 4);
    for y in 1..3 {
        for x in 1..3 {
            mask.put_pixel(x, y, Rgba::WHITE.to_image_rgba());
        }
    }

    clear_map_outside_local_mask(&mut img, &layout, &mask, bg);

    assert_eq!(img.get_pixel(1, 1).0, bg.to_image_rgba().0);
    assert_eq!(img.get_pixel(2, 2).0, Rgba::BLACK.to_image_rgba().0);
}

#[test]
fn trim_vertical_canvas_whitespace_crops_outer_blank_rows() {
    let mut img = RgbaImage::from_pixel(
        6,
        10,
        RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology)
            .canvas_background
            .to_image_rgba(),
    );
    for y in 3..7 {
        for x in 0..6 {
            img.put_pixel(x, y, Rgba::BLACK.to_image_rgba());
        }
    }

    let (trimmed, crop_top) = trim_vertical_canvas_whitespace(
        &img,
        RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology).canvas_background,
    );

    assert_eq!(trimmed.width(), 6);
    assert!(trimmed.height() < 10);
    assert!(trimmed.height() >= 4);
    // Content starts at row 3 and the pass keeps a 2-row pad, so exactly
    // one row came off the top -- the offset the plot rect follows.
    assert_eq!(crop_top, 1);
}

#[test]
fn center_horizontal_canvas_content_balances_outer_margins() {
    let bg = RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology).canvas_background;
    let mut img = RgbaImage::from_pixel(12, 4, bg.to_image_rgba());
    for x in 1..7 {
        img.put_pixel(x, 1, Rgba::BLACK.to_image_rgba());
    }

    let (centered, shift) = center_horizontal_canvas_content(&img, bg);
    let mut min_x = centered.width();
    let mut max_x = 0;
    for y in 0..centered.height() {
        for x in 0..centered.width() {
            if !pixel_matches_background(*centered.get_pixel(x, y), bg) {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
            }
        }
    }
    let left_margin = min_x;
    let right_margin = centered.width().saturating_sub(max_x).saturating_sub(1);

    assert!(left_margin.abs_diff(right_margin) <= 1);
    // Content spans x 1..=6 in a 12-wide canvas: margins 1 and 5, so the
    // pass moves everything 2 px right and must SAY so.
    assert_eq!(shift, 2);
}

#[test]
fn crop_canvas_whitespace_removes_blank_border() {
    let bg = RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology).canvas_background;
    let mut img = RgbaImage::from_pixel(12, 10, bg.to_image_rgba());
    for y in 3..7 {
        for x in 4..9 {
            img.put_pixel(x, y, Rgba::BLACK.to_image_rgba());
        }
    }

    let (cropped, (crop_left, crop_top)) = crop_canvas_whitespace(&img, bg, 1);

    assert_eq!(cropped.width(), 7);
    assert_eq!(cropped.height(), 6);
    // Content starts at (4, 3) and the pass keeps a 1-px pad: 3 columns
    // and 2 rows came off -- the offsets the plot rect follows.
    assert_eq!((crop_left, crop_top), (3, 2));
}

#[test]
fn bucketed_contours_match_legacy_when_projected_corner_is_missing() {
    let layout = contour_test_layout();
    let overlay = ContourOverlay {
        data: vec![0.0, 1.0, 2.0, 3.0],
        ny: 2,
        nx: 2,
        levels: vec![1.5],
        color: Rgba::BLACK,
        width: 1,
        labels: false,
        show_extrema: false,
        pattern: crate::request::ContourLinePattern::Solid,
        major_every: None,
        major_width: None,
    };
    let pixel_points = vec![
        Some((0.0, 0.0)),
        None,
        Some((64.0, 64.0)),
        Some((0.0, 64.0)),
    ];

    let mut legacy = blank_test_image();
    let mut bucketed = blank_test_image();
    let mut legacy_labels = ContourLabelPlacer::default();
    let mut bucketed_labels = ContourLabelPlacer::default();
    draw_contours_legacy(
        &mut legacy,
        &layout,
        &overlay,
        Some(&pixel_points),
        None,
        &mut legacy_labels,
        1,
    );
    draw_contours_bucketed(
        &mut bucketed,
        &layout,
        &overlay,
        Some(&pixel_points),
        None,
        &mut bucketed_labels,
        1,
    );

    assert_eq!(legacy, bucketed);
}

#[test]
fn contour_cells_reject_projected_seam_jumps() {
    let layout = contour_test_layout();
    let overlay = ContourOverlay {
        data: vec![0.0, 1.0, 2.0, 3.0],
        ny: 2,
        nx: 2,
        levels: vec![1.5],
        color: Rgba::BLACK,
        width: 1,
        labels: false,
        show_extrema: false,
        pattern: crate::request::ContourLinePattern::Solid,
        major_every: None,
        major_width: None,
    };
    let pixel_points = vec![
        Some((0.0, 0.0)),
        Some((1000.0, 0.0)),
        Some((0.0, 64.0)),
        Some((1000.0, 64.0)),
    ];

    assert!(contour_cell_corners(&layout, &overlay, Some(&pixel_points), 0, 1).is_none());
}

#[test]
fn projected_pixel_bilinear_rejects_projected_seam_jumps() {
    let pixel_points = vec![
        Some((0.0, 0.0)),
        Some((1000.0, 0.0)),
        Some((0.0, 64.0)),
        Some((1000.0, 64.0)),
    ];

    assert!(projected_pixel_bilinear(&pixel_points, 2, 2, 0.5, 0.5).is_none());
}

#[test]
fn levels_are_sorted_finite_rejects_unsorted_or_nan_levels() {
    assert!(levels_are_sorted_finite(&[1.0, 2.0, 3.0]));
    assert!(!levels_are_sorted_finite(&[2.0, 1.0]));
    assert!(!levels_are_sorted_finite(&[1.0, f64::NAN, 3.0]));
}

#[test]
fn render_to_png_reuses_projected_pixel_cache_for_identical_meshes() {
    let _guard = PROJECTED_PIXEL_CACHE_TEST_LOCK.lock().unwrap();
    reset_projected_pixel_cache_for_tests();

    let data = [0.0, 1.0, 2.0, 3.0];
    let opts = sample_projected_opts();

    let first = render_to_png(&data, 2, 2, &opts);
    let second = render_to_png(&data, 2, 2, &opts);

    assert_eq!(first, second);
    assert_eq!(projected_pixel_cache_miss_count_for_tests(), 1);
}

#[test]
fn render_to_png_recomputes_projected_pixels_when_extent_changes() {
    let _guard = PROJECTED_PIXEL_CACHE_TEST_LOCK.lock().unwrap();
    reset_projected_pixel_cache_for_tests();

    let data = [0.0, 1.0, 2.0, 3.0];
    let opts = sample_projected_opts();
    let mut shifted = sample_projected_opts();
    shifted.map_extent = Some(MapExtent {
        x_min: -0.25,
        x_max: 0.75,
        y_min: 0.0,
        y_max: 1.0,
    });

    render_to_png(&data, 2, 2, &opts);
    render_to_png(&data, 2, 2, &shifted);

    assert_eq!(projected_pixel_cache_miss_count_for_tests(), 2);
}

#[test]
fn static_base_cache_key_changes_with_plot_style() {
    let opts = sample_projected_opts();
    let layout = compute_layout(
        opts.width,
        opts.height,
        opts.colorbar,
        opts.title.is_some(),
        opts.presentation,
        opts.chrome_scale,
    );
    let baseline_key = static_base_cache_key(
        &opts,
        &layout,
        opts.map_extent.as_ref(),
        None,
        opts.presentation.canvas_background,
        opts.presentation.map_background,
        true,
    );
    let mut clean_opts = opts.clone();
    clean_opts.presentation = RenderPresentation::for_mode_with_style(
        ProductVisualMode::FilledMeteorology,
        crate::presentation::StaticPlotStyle::CleanAtlasFast,
    );
    let clean_key = static_base_cache_key(
        &clean_opts,
        &layout,
        clean_opts.map_extent.as_ref(),
        None,
        clean_opts.presentation.canvas_background,
        clean_opts.presentation.map_background,
        true,
    );

    assert_ne!(baseline_key, clean_key);
}

#[test]
fn map_frame_aspect_ratio_matches_wide_render_layout() {
    let default_layout = compute_layout(
        1200,
        900,
        true,
        true,
        RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology),
        ChromeScale::default(),
    );
    let ratio = default_layout.map_w as f64 / default_layout.map_h as f64;
    assert!(ratio > 1.35);
    assert!(ratio < 1.7);

    let operational_layout = compute_layout(
        1200,
        900,
        true,
        true,
        RenderPresentation::for_mode_with_style(
            ProductVisualMode::FilledMeteorology,
            crate::presentation::StaticPlotStyle::OperationalFast,
        ),
        ChromeScale::default(),
    );
    let operational_ratio = operational_layout.map_w as f64 / operational_layout.map_h as f64;
    assert!(operational_ratio > 1.2);
    assert!(operational_ratio < ratio);
}

#[test]
fn colorbar_tick_levels_follow_legend_levels_when_fill_is_densified() {
    let cmap = LeveledColormap::from_palette_with_options(
        &[Rgba::new(0, 0, 255), Rgba::new(255, 0, 0)],
        &[0.0, 10.0, 20.0, 30.0, 40.0],
        Extend::Neither,
        None,
        ColormapBuildOptions {
            render_density: crate::colormap::RenderDensity::default(),
            legend: crate::colormap::LegendControls {
                density: LevelDensity::default(),
                mode: crate::colormap::LegendMode::Stepped,
            },
        },
    );

    assert!(cmap.levels.len() > cmap.legend_levels.len());
    assert_eq!(
        colorbar_levels_for_ticks(&cmap),
        cmap.legend_levels.as_slice()
    );
}

#[test]
fn colorbar_tick_labels_clamp_to_requested_bounds() {
    let labels =
        filter_tick_labels_to_fit(&[0.0, 50.0, 100.0], 0.0, 100.0, 80, 120, 80, 200, 400, 1);
    assert!(!labels.is_empty());
    for (_, lx, label) in labels {
        let width = text::text_width(&label, 1) as i32;
        assert!(lx >= 80);
        assert!(lx + width <= 200);
    }
}

#[test]
fn extrema_selection_keeps_only_ranked_spaced_centers() {
    let mut layout = contour_test_layout();
    layout.map_w = 1500;
    layout.map_h = 850;
    let lows = vec![
        ExtremaCandidate {
            value: 1009.8,
            score: 1009.8,
            px: 120,
            py: 120,
        },
        ExtremaCandidate {
            value: 1008.2,
            score: 1008.2,
            px: 170,
            py: 145,
        },
        ExtremaCandidate {
            value: 1006.5,
            score: 1006.5,
            px: 760,
            py: 430,
        },
        ExtremaCandidate {
            value: 1004.1,
            score: 1004.1,
            px: 1260,
            py: 240,
        },
        ExtremaCandidate {
            value: 1003.9,
            score: 1003.9,
            px: 1320,
            py: 260,
        },
    ];

    let selected = select_extrema_labels(lows, false, &layout);

    assert_eq!(selected.len(), 3);
    assert!(selected.iter().any(|point| point.value == 1003.9));
    assert!(selected.iter().any(|point| point.value == 1006.5));
    assert!(!selected.iter().any(|point| point.value == 1004.1));
    assert!(!selected.iter().any(|point| point.value == 1009.8));
}

#[test]
fn contour_labels_scale_up_on_operational_sized_maps() {
    let small = contour_test_layout();
    let mut operational = contour_test_layout();
    operational.map_w = 1494;
    operational.map_h = 829;

    assert_eq!(contour_label_scale(&small), 1);
    assert_eq!(contour_label_scale(&operational), 2);
    assert_eq!(contour_label_halo_width(&operational), 2);
    assert!(contour_label_size_factor(&operational) < 1.0);
}

#[test]
fn extrema_analysis_grid_keeps_full_resolution_by_default() {
    assert_eq!(extrema_analysis_stride(100, 100), 1);
    assert_eq!(extrema_analysis_stride(1800, 1059), 1);

    let data = (0..36).map(|value| value as f64).collect::<Vec<_>>();
    let (analysis, nx, ny) = extrema_analysis_grid(&data, 6, 6, 2);

    assert_eq!((nx, ny), (3, 3));
    assert_eq!(analysis.len(), 9);
    assert!((analysis[0] - 3.5).abs() < 1.0e-9);
    assert!((analysis[8] - 31.5).abs() < 1.0e-9);
}

#[test]
fn chrome_scale_grows_layout_for_larger_outputs() {
    let base = compute_layout(
        1200,
        900,
        true,
        true,
        RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology),
        ChromeScale::default(),
    );
    let bigger = compute_layout(
        2400,
        1800,
        true,
        true,
        RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology),
        ChromeScale::default(),
    );

    assert!(bigger.cbar_h > base.cbar_h);
    assert!(bigger.text_scale > base.text_scale);
    assert!(bigger.label_gap > base.label_gap);
}

#[test]
fn filled_layout_keeps_header_and_legend_tight_to_map() {
    let layout = compute_layout(
        1200,
        900,
        true,
        true,
        RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology),
        ChromeScale::Fixed(1.0),
    );

    assert_eq!(layout.map_y, 64);
    assert_eq!(layout.title_y, 5);
    assert!(layout.subtitle_y > layout.title_y);
    assert_eq!(layout.cbar_y + layout.cbar_h, 892);
}

#[test]
fn render_to_png_suppresses_barbs_when_overlay_data_is_nan() {
    // Updated expectation: barb overlays are no longer clipped to the fill
    // raster (that broke height-contour / wind-barb renders when the fill
    // used mask_below). Instead, barbs clip themselves via NaN u/v values.
    let _guard = PROJECTED_PIXEL_CACHE_TEST_LOCK.lock().unwrap();
    let mut opts = sample_projected_opts();
    opts.title = None;
    opts.barbs = vec![BarbOverlay {
        u: vec![f32::NAN; 4],
        v: vec![f32::NAN; 4],
        ny: 2,
        nx: 2,
        stride_x: 1,
        stride_y: 1,
        spacing_px: 24.0,
        color: Rgba::BLACK,
        halo_color: Rgba::WHITE,
        halo_width: 2,
        width: 1,
        length_px: 12.0,
    }];

    let data = [0.5f64; 4];
    let png = render_to_png(&data, 2, 2, &opts);
    let image = image::load_from_memory_with_format(&png, image::ImageFormat::Png)
        .unwrap()
        .to_rgba8();
    // NaN u/v means no barb glyphs are drawn.
    let non_fill = image.pixels().filter(|px| px.0 == [0, 0, 0, 255]).count();
    assert_eq!(non_fill, 0, "NaN barb vectors should produce no glyphs");
}

#[test]
fn barb_glyph_margin_skips_map_edge_anchors() {
    assert!(
        !barb_glyph_fits_map_rect(10.0, 10.0, 100, 100, 18.0, 1),
        "edge anchors can draw outside the map frame"
    );
    assert!(
        barb_glyph_fits_map_rect(50.0, 50.0, 100, 100, 18.0, 1),
        "center anchors should still render"
    );
}

#[test]
fn render_to_png_suppresses_contours_when_overlay_data_is_nan() {
    // Updated expectation: contour overlays self-clip via NaN data, not via
    // the fill raster. Lets height contours render across the whole frame
    // even when the paired CAPE fill uses mask_below.
    let _guard = PROJECTED_PIXEL_CACHE_TEST_LOCK.lock().unwrap();
    let mut opts = sample_projected_opts();
    opts.title = None;
    opts.contours = vec![ContourOverlay {
        data: vec![f64::NAN; 4],
        ny: 2,
        nx: 2,
        levels: vec![1.5],
        color: Rgba::BLACK,
        width: 1,
        labels: false,
        show_extrema: false,
        pattern: crate::request::ContourLinePattern::Solid,
        major_every: None,
        major_width: None,
    }];

    let data = [0.5f64; 4];
    let png = render_to_png(&data, 2, 2, &opts);
    let image = image::load_from_memory_with_format(&png, image::ImageFormat::Png)
        .unwrap()
        .to_rgba8();
    let contour_pixels = image.pixels().filter(|px| px.0 == [0, 0, 0, 255]).count();
    assert_eq!(
        contour_pixels, 0,
        "NaN contour data should produce no contour lines"
    );
}

#[test]
fn crates_do_not_reintroduce_legacy_credit_footers() {
    let crates_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let forbidden = [
        String::from_utf8(vec![
            67, 111, 108, 111, 114, 32, 84, 97, 98, 108, 101, 115, 58, 32, 83, 111, 108, 97, 114,
            112, 111, 119, 101, 114, 48, 55,
        ])
        .expect("legacy footer bytes should be valid utf-8"),
        ["Pivotal", " Weather"].concat(),
        ["Weather", "Bell"].concat(),
    ];
    let mut offenders = Vec::<String>::new();
    visit_rs_files(&crates_root, &mut |path| {
        if let Ok(contents) = std::fs::read_to_string(path) {
            for term in &forbidden {
                if contents.contains(term) {
                    offenders.push(format!("{} => {}", path.display(), term));
                }
            }
        }
    })
    .expect("crate source tree should be readable");
    assert!(
        offenders.is_empty(),
        "legacy credit/footer strings remain in crates/: {offenders:?}"
    );
}


#[test]
fn subtitle_left_keeps_what_the_source_label_does_not_need() {
    // B-13.  The split was `subtitle_available / 2` whenever a right
    // subtitle existed, so the provenance line was ellipsized to half
    // the row while "source: ArWen" -- a small fraction of it -- left the
    // rest of the row empty.  Measured on a rendered plot the line came
    // out `Init 04/03 12Z | F006 | Valid 04/03 18Z | WRF...`, and what
    // was dropped was the grid spacing.
    let scale = 2u32;
    let gap = 6u32 * scale;
    let right = "source: ArWen";
    let provenance = "Init 04/03 12Z | F006 | Valid 04/03 18Z | WRF | dx 12 km";

    // Widths are measured, not guessed: the row is exactly wide enough
    // for both, which is the case the old rule got wrong.
    let prov_w = measure_text_width(provenance, scale, false);
    let right_w = measure_text_width(right, scale, false);
    let available = prov_w + right_w + gap;

    let left_width = subtitle_left_width(available, Some(right), scale, gap);
    assert_eq!(left_width, prov_w);
    let (fitted, truncated) = fit_text_to_width(provenance, left_width, scale, false);
    assert!(!truncated, "still truncated: {fitted:?}");
    assert_eq!(fitted, provenance);

    // Negative control: the old fixed half-split cuts the same line, so
    // this test fails if the split ever goes back to it.
    assert!(available / 2 < prov_w, "fixture cannot demonstrate B-13");
    let (old_fitted, old_truncated) =
        fit_text_to_width(provenance, available / 2, scale, false);
    assert!(old_truncated);
    assert!(old_fitted.ends_with("..."));
}

#[test]
fn subtitle_left_never_falls_below_the_even_split() {
    // A pathological source label must not squeeze the provenance below
    // what the old fixed split gave it.
    let scale = 2u32;
    let available = 400u32;
    let absurd = "source: ".to_string() + &"x".repeat(400);
    let left_width = subtitle_left_width(available, Some(absurd.as_str()), scale, 12);
    assert_eq!(left_width, available - available / 2);
}

#[test]
fn a_subtitle_that_does_not_fit_reports_that_it_was_cut() {
    // The channel B-13 said was missing: the plain helper returns a bare
    // String, so a caller could not tell a line that fitted from one
    // that lost its tail.
    let scale = 2u32;
    let text = "Init 04/03 12Z | F006 | Valid 04/03 18Z | WRF | dx 12 km";
    let (fitted, truncated) = fit_text_to_width(text, 60, scale, false);
    assert!(truncated);
    assert!(fitted.ends_with("..."));
    let (whole, untruncated) = fit_text_to_width(text, 100_000, scale, false);
    assert!(!untruncated);
    assert_eq!(whole, text);
}

/// A regular latitude/longitude mesh, projected into the presentation
/// projection the direct lane picks for a CONUS-sized window.
///
/// This is the geometry every lat/lon SOURCE renders in -- GFS, GDAS,
/// GEFS, ICON, the AI atmospheres -- once a regional window makes the
/// adaptive presentation conic.  Its screen footprint is a curved
/// quadrilateral, not a rectangle.
fn curved_latlon_footprint(map_w: u32, map_h: u32) -> (ProjectedGrid, Vec<Option<(f32, f32)>>) {
    let (ny, nx) = (36usize, 71usize);
    let projector = crate::projection::ProjectionSpec::LambertConformal {
        standard_parallel_1_deg: 25.833_333_333_333_332,
        standard_parallel_2_deg: 49.166_666_666_666_664,
        central_meridian_deg: -95.0,
    }
    .build_projector(Some(39.0), None, &[20.0f32], &[-95.0f32])
    .expect("lambert projector");

    let mut x = Vec::with_capacity(ny * nx);
    let mut y = Vec::with_capacity(ny * nx);
    for j in 0..ny {
        for i in 0..nx {
            let lat = 20.0 + j as f64;
            let lon = -130.0 + i as f64;
            let (px, py) = projector.project(lat, lon);
            x.push(px);
            y.push(py);
        }
    }
    let (mut x0, mut x1, mut y0, mut y1) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for (&px, &py) in x.iter().zip(y.iter()) {
        x0 = x0.min(px);
        x1 = x1.max(px);
        y0 = y0.min(py);
        y1 = y1.max(py);
    }
    let extent = MapExtent {
        x_min: x0,
        x_max: x1,
        y_min: y0,
        y_max: y1,
    };
    let pixel_points = x
        .iter()
        .zip(y.iter())
        .map(|(&px, &py)| {
            extent
                .to_pixel(px, py, map_w, map_h)
                .map(|(a, b)| (a as f32, b as f32))
        })
        .collect::<Vec<_>>();
    (ProjectedGrid { x, y, ny, nx }, pixel_points)
}

/// A native projected grid -- a wrfout on its own Lambert -- is an
/// axis-aligned rectangle on screen.
fn rectangular_footprint(map_w: u32, map_h: u32) -> (ProjectedGrid, Vec<Option<(f32, f32)>>) {
    let (ny, nx) = (36usize, 71usize);
    let mut x = Vec::with_capacity(ny * nx);
    let mut y = Vec::with_capacity(ny * nx);
    for j in 0..ny {
        for i in 0..nx {
            x.push(i as f64);
            y.push(j as f64);
        }
    }
    let extent = MapExtent {
        x_min: 0.0,
        x_max: (nx - 1) as f64,
        y_min: 0.0,
        y_max: (ny - 1) as f64,
    };
    let pixel_points = x
        .iter()
        .zip(y.iter())
        .map(|(&px, &py)| {
            extent
                .to_pixel(px, py, map_w, map_h)
                .map(|(a, b)| (a as f32, b as f32))
        })
        .collect::<Vec<_>>();
    (ProjectedGrid { x, y, ny, nx }, pixel_points)
}

fn covered_pixel_bounds(
    grid: &ProjectedGrid,
    pixels: &[Option<(f32, f32)>],
    w: u32,
    h: u32,
) -> LocalRect {
    let mask = crate::rasterize::rasterize_projected_coverage_mask(grid.ny, grid.nx, pixels, w, h);
    LocalRect::from_bounds(raster_alpha_bounds(&mask).expect("some coverage"))
}

#[test]
fn resampled_lat_lon_domain_frame_holds_every_drawn_row() {
    // The defect: `inner_rect_from_coverage` returns the largest
    // rectangle INSCRIBED in the coverage, which for a curved footprint
    // is a thin horizontal band -- and `clear_outside` then erases the
    // rest of the domain.  The aigfs lane saw it as "rw_wrfbatch draws
    // only about half the south_north rows".
    let (map_w, map_h) = (1200u32, 800u32);
    let (grid, pixels) = curved_latlon_footprint(map_w, map_h);
    let frame = DomainFrame {
        inset_px: 2,
        outline_width: 2,
        source: DomainFrameSource::ProjectedGrid,
        ..DomainFrame::map_viewport_default()
    };

    let covered = covered_pixel_bounds(&grid, &pixels, map_w, map_h);

    // The mechanism, pinned so it cannot come back by accident: asked to
    // INSCRIBE, the same coverage yields a band a fraction of the domain
    // tall.  This is the arm that shipped, and what it drops is model
    // data that `clear_outside` then erases.
    let inscribed =
        compute_projected_domain_frame_rect(frame, &grid, &pixels, map_w, map_h, 0, true)
            .expect("a covered domain has a frame");
    assert!(
        inscribed.height() * 2 < covered.height(),
        "the inscribed rectangle of a curved footprint is a band: {} of \
         {} rows",
        inscribed.height(),
        covered.height()
    );

    let rect = compute_projected_domain_frame_rect(frame, &grid, &pixels, map_w, map_h, 0, false)
        .expect("a covered domain has a frame");

    assert_eq!(
        rect,
        inset_rect(covered, frame.inset_px).expect("inset fits"),
        "a resampled lat/lon field's frame is the box that holds every \
         drawn row, not the widest band inscribed inside the curve"
    );
    assert!(
        rect.height() * 100 >= covered.height() * 95,
        "frame kept {} of {} covered rows",
        rect.height(),
        covered.height()
    );
}

#[test]
fn native_projected_domain_frame_still_inscribes_the_rectangle() {
    // The unchanged half of the contract: a wrfout on its own projection
    // is a screen rectangle, so inscribing and bounding agree, and the
    // inscribing path stays exactly where it was.
    let (map_w, map_h) = (1200u32, 800u32);
    let (grid, pixels) = rectangular_footprint(map_w, map_h);
    let frame = DomainFrame {
        inset_px: 2,
        outline_width: 2,
        source: DomainFrameSource::ProjectedGrid,
        ..DomainFrame::map_viewport_default()
    };

    let inscribed =
        compute_projected_domain_frame_rect(frame, &grid, &pixels, map_w, map_h, 0, true)
            .expect("frame");
    let bounded =
        compute_projected_domain_frame_rect(frame, &grid, &pixels, map_w, map_h, 0, false)
            .expect("frame");

    assert_eq!(inscribed, bounded);
}
