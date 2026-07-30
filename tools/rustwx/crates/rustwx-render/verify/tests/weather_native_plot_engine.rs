use rustwx_render_verify::{
    Color, PanelGridLayout, RgbaImage, WeatherProduct, count_non_background_pixels,
    default_output_dir, render_image, render_panel_grid, sample_contour_fill_alignment_request,
    sample_overlay_request, sample_panel_requests, sample_projected_contour_request,
    sample_weather_request, save_png,
};
use std::error::Error;

fn is_white(pixel: [u8; 4]) -> bool {
    pixel[3] > 0 && pixel[0] >= 250 && pixel[1] >= 250 && pixel[2] >= 250
}

fn is_dark(pixel: [u8; 4]) -> bool {
    pixel[3] > 180 && (pixel[0] as u16 + pixel[1] as u16 + pixel[2] as u16) <= 170
}

fn color_distance(a: [u8; 4], b: [u8; 4]) -> u32 {
    (a[0] as i32 - b[0] as i32).unsigned_abs()
        + (a[1] as i32 - b[1] as i32).unsigned_abs()
        + (a[2] as i32 - b[2] as i32).unsigned_abs()
}

fn non_white_bbox(image: &RgbaImage) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = u32::MAX;
    let mut max_x = 0;
    let mut min_y = u32::MAX;
    let mut max_y = 0;

    for (x, y, pixel) in image.enumerate_pixels() {
        if is_white(pixel.0) {
            continue;
        }
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    (min_x != u32::MAX).then_some((min_x, max_x, min_y, max_y))
}

fn dark_column_clusters(image: &RgbaImage, min_dark_pixels: usize) -> Vec<u32> {
    let counts: Vec<usize> = (0..image.width())
        .map(|x| {
            (0..image.height())
                .filter(|&y| is_dark(image.get_pixel(x, y).0))
                .count()
        })
        .collect();

    let mut clusters = Vec::new();
    let mut x = 0usize;
    while x < counts.len() {
        if counts[x] < min_dark_pixels {
            x += 1;
            continue;
        }

        let start = x;
        let mut best_x = x;
        let mut best_count = counts[x];
        while x + 1 < counts.len() && counts[x + 1] >= min_dark_pixels {
            x += 1;
            if counts[x] > best_count {
                best_count = counts[x];
                best_x = x;
            }
        }

        if x >= start {
            clusters.push(best_x as u32);
        }
        x += 1;
    }

    clusters
}

fn average_fill_window_color(
    image: &RgbaImage,
    center_x: i32,
    center_y: i32,
    radius_x: i32,
    radius_y: i32,
) -> Option<[u8; 4]> {
    let mut count = 0u32;
    let mut r = 0u32;
    let mut g = 0u32;
    let mut b = 0u32;

    for y in (center_y - radius_y)..=(center_y + radius_y) {
        if y < 0 || y >= image.height() as i32 {
            continue;
        }
        for x in (center_x - radius_x)..=(center_x + radius_x) {
            if x < 0 || x >= image.width() as i32 {
                continue;
            }
            let pixel = image.get_pixel(x as u32, y as u32).0;
            if is_white(pixel) || is_dark(pixel) {
                continue;
            }
            count += 1;
            r += pixel[0] as u32;
            g += pixel[1] as u32;
            b += pixel[2] as u32;
        }
    }

    (count > 0).then_some([(r / count) as u8, (g / count) as u8, (b / count) as u8, 255])
}

fn projected_contour_support_ratio(image: &RgbaImage) -> (usize, f64, (u32, u32, u32, u32)) {
    let mut supported = 0usize;
    let mut total = 0usize;
    let mut min_x = u32::MAX;
    let mut max_x = 0;
    let mut min_y = u32::MAX;
    let mut max_y = 0;

    for y in 2..image.height().saturating_sub(2) {
        for x in 2..image.width().saturating_sub(2) {
            if !is_dark(image.get_pixel(x, y).0) {
                continue;
            }
            total += 1;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);

            let mut nearby_fill = 0usize;
            for yy in (y - 2)..=(y + 2) {
                for xx in (x - 2)..=(x + 2) {
                    if xx == x && yy == y {
                        continue;
                    }
                    let pixel = image.get_pixel(xx, yy).0;
                    if !is_white(pixel) && !is_dark(pixel) {
                        nearby_fill += 1;
                    }
                }
            }
            if nearby_fill >= 4 {
                supported += 1;
            }
        }
    }

    let bbox = if min_x == u32::MAX {
        (0, 0, 0, 0)
    } else {
        (min_x, max_x, min_y, max_y)
    };
    let ratio = if total == 0 {
        0.0
    } else {
        supported as f64 / total as f64
    };
    (total, ratio, bbox)
}

#[test]
fn filled_weather_native_request_renders_visible_content() -> Result<(), Box<dyn Error>> {
    let request = sample_weather_request(WeatherProduct::Sbecape)?;
    let image = render_image(&request)?;

    assert_eq!(image.width(), request.width);
    assert_eq!(image.height(), request.height);
    assert!(
        count_non_background_pixels(&image, [255, 255, 255, 255]) > 12_000,
        "filled request should draw substantial plot content"
    );
    Ok(())
}

#[test]
fn overlay_weather_native_request_renders_contours_and_barbs() -> Result<(), Box<dyn Error>> {
    let request = sample_overlay_request()?;
    let image = render_image(&request)?;

    assert_eq!(image.width(), request.width);
    assert_eq!(image.height(), request.height);
    assert!(
        count_non_background_pixels(&image, [255, 255, 255, 255]) > 3_500,
        "overlay request should draw visible contours and barbs"
    );
    Ok(())
}

#[test]
fn contour_lines_track_discrete_fill_boundaries() -> Result<(), Box<dyn Error>> {
    let request = sample_contour_fill_alignment_request()?;
    let image = render_image(&request)?;
    let mut fill_only = request.clone();
    fill_only.contours.clear();
    let fill_only_image = render_image(&fill_only)?;
    let (min_x, max_x, min_y, max_y) = non_white_bbox(&fill_only_image)
        .expect("fill+contour alignment render should contain visible ink");
    let fill_width = max_x.saturating_sub(min_x).saturating_add(1);
    let row = min_y + max_y.saturating_sub(min_y) / 2;
    let contour_columns = dark_column_clusters(&image, (image.height() / 4) as usize);

    assert_eq!(
        contour_columns.len(),
        3,
        "expected one contour cluster per internal fill boundary"
    );

    for (index, &column) in contour_columns.iter().enumerate() {
        let expected = min_x as f64 + (fill_width as f64 * ((index + 1) as f64 / 4.0));
        assert!(
            (column as i32 - expected.round() as i32).unsigned_abs() <= 14,
            "contour line {index} should sit near the {}/4 fill boundary; got column {}, expected about {}",
            index + 1,
            column,
            expected.round() as i32
        );
        let left_band_center = min_x as f64 + (fill_width as f64 * ((index as f64 + 0.5) / 4.0));
        let right_band_center = min_x as f64 + (fill_width as f64 * ((index as f64 + 1.5) / 4.0));
        let left = average_fill_window_color(
            &fill_only_image,
            left_band_center.round() as i32,
            row as i32,
            6,
            18,
        )
        .expect("left fill band should remain visible");
        let right = average_fill_window_color(
            &fill_only_image,
            right_band_center.round() as i32,
            row as i32,
            6,
            18,
        )
        .expect("right fill band should remain visible");
        assert!(
            color_distance(left, right) >= 60,
            "contour line {index} should separate clearly different fill bands"
        );
    }

    Ok(())
}

#[test]
fn projected_contours_remain_supported_by_projected_fill_footprint() -> Result<(), Box<dyn Error>> {
    let request = sample_projected_contour_request()?;
    let image = render_image(&request)?;
    let (dark_pixels, support_ratio, (min_x, max_x, min_y, max_y)) =
        projected_contour_support_ratio(&image);

    assert!(
        dark_pixels > 500,
        "projected contour verification render should contain substantial contour ink"
    );
    assert!(
        support_ratio >= 0.93,
        "projected contour pixels should stay attached to projected fill coverage; support ratio={support_ratio:.3}"
    );
    assert!(
        max_x.saturating_sub(min_x) > image.width() / 4,
        "projected contour path should span a meaningful horizontal slice of the domain"
    );
    assert!(
        max_y.saturating_sub(min_y) > image.height() / 2,
        "projected contour path should span a meaningful vertical slice of the domain"
    );

    Ok(())
}

#[test]
fn mixed_panel_requests_render_weather_native_smoke_grid() -> Result<(), Box<dyn Error>> {
    let requests = sample_panel_requests()?;
    let layout =
        PanelGridLayout::two_by_two(320, 220)?.with_background(Color::rgba(244, 244, 244, 255));
    let image = render_panel_grid(&layout, &requests)?;

    assert_eq!(image.width(), 640);
    assert_eq!(image.height(), 440);
    assert!(
        count_non_background_pixels(&image, [244, 244, 244, 255]) > 45_000,
        "panel render should contain multiple populated panes"
    );
    Ok(())
}

#[test]
fn verify_harness_can_write_png_artifacts() -> Result<(), Box<dyn Error>> {
    let request = sample_weather_request(WeatherProduct::Scp)?;
    let output_dir = default_output_dir();
    std::fs::create_dir_all(&output_dir)?;
    let output_path = output_dir.join(format!("verify-write-smoke-{}.png", std::process::id()));

    save_png(&request, &output_path)?;
    let bytes = std::fs::read(&output_path)?;
    assert!(bytes.starts_with(&[137, 80, 78, 71, 13, 10, 26, 10]));

    let _ = std::fs::remove_file(output_path);
    Ok(())
}
