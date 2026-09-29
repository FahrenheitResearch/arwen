use super::*;

thread_local! {
    static EDGE_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) fn record_edge_visit() {
    EDGE_VISITS.with(|visits| visits.set(visits.get() + 1));
}

#[test]
fn detailed_polygon_scan_has_bounded_edge_visits() {
    let rings = vec![detailed_ring(32_768, 480.0, 360.0)];
    let mut actual = RgbaImage::new(480, 360);
    let mut expected = actual.clone();
    EDGE_VISITS.with(|visits| visits.set(0));
    fill_polygon(&mut actual, &rings, Rgba::BLACK, None);
    let visits = EDGE_VISITS.with(|visits| visits.get());
    reference_fill(&mut expected, &rings, Rgba::BLACK, None);
    assert_eq!(actual, expected);
    // Count edge visits instead of timing the machine. The old scan visits
    // about ten million edges for this image, including inactive edges.
    assert!(
        visits > 0,
        "the production scan must report its edge visits"
    );
    assert!(visits < 100_000, "too many row/edge tests: {visits}");
}

fn reference_fill(
    img: &mut RgbaImage,
    rings: &[Vec<(f64, f64)>],
    color: Rgba,
    clip: Option<(i32, i32, i32, i32)>,
) {
    if rings.is_empty() || color.a == 0 {
        return;
    }

    let img_w = img.width() as i32;
    let img_h = img.height() as i32;
    let (cx0, cy0, cx1, cy1) = match clip {
        Some((x0, y0, x1, y1)) => (x0.max(0), y0.max(0), x1.min(img_w - 1), y1.min(img_h - 1)),
        None => (0, 0, img_w - 1, img_h - 1),
    };
    if cx1 < cx0 || cy1 < cy0 {
        return;
    }

    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for ring in rings {
        for &(_, y) in ring {
            if y.is_finite() {
                y_min = y_min.min(y);
                y_max = y_max.max(y);
            }
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() || y_max < cy0 as f64 {
        return;
    }
    let y0 = y_min.floor().max(cy0 as f64) as i32;
    let y1 = (y_max.ceil() as i32).min(cy1);
    if y1 < y0 {
        return;
    }

    #[derive(Clone)]
    struct Edge {
        y_min: f64,
        y_max: f64,
        x: f64,
        dx: f64,
    }
    let mut edges: Vec<Edge> = Vec::new();
    for ring in rings {
        let n = ring.len();
        if n < 2 {
            continue;
        }
        for i in 0..n {
            let (ax, ay) = ring[i];
            let (bx, by) = ring[(i + 1) % n];
            if !ax.is_finite() || !ay.is_finite() || !bx.is_finite() || !by.is_finite() {
                continue;
            }
            if (ay - by).abs() < 1e-9 {
                continue; // horizontal edges contribute nothing to even-odd
            }
            let (lo_y, hi_y, lo_x, hi_x) = if ay < by {
                (ay, by, ax, bx)
            } else {
                (by, ay, bx, ax)
            };
            let dx = (hi_x - lo_x) / (hi_y - lo_y);
            edges.push(Edge {
                y_min: lo_y,
                y_max: hi_y,
                x: lo_x,
                dx,
            });
        }
    }
    if edges.is_empty() {
        return;
    }

    let mut xs: Vec<f64> = Vec::with_capacity(edges.len());
    let opaque_fill = color.a == 255;
    let opaque_rgba = color.to_image_rgba();
    for y in y0..=y1 {
        let yf = y as f64 + 0.5;
        xs.clear();
        for edge in &edges {
            if yf >= edge.y_min && yf < edge.y_max {
                xs.push(edge.x + (yf - edge.y_min) * edge.dx);
            }
        }
        if xs.len() < 2 {
            continue;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut i = 0;
        while i + 1 < xs.len() {
            let xa = xs[i].max(cx0 as f64).ceil() as i32;
            let xb = xs[i + 1].min(cx1 as f64).floor() as i32;
            if xb >= xa {
                if opaque_fill {
                    for x in xa..=xb {
                        img.put_pixel(x as u32, y as u32, opaque_rgba);
                    }
                } else {
                    for x in xa..=xb {
                        blend_pixel(img, x, y, color);
                    }
                }
            }
            i += 2;
        }
    }
}

fn detailed_ring(count: usize, width: f64, height: f64) -> Vec<(f64, f64)> {
    (0..count)
        .map(|i| {
            let angle = i as f64 * std::f64::consts::TAU / count as f64;
            let radius = 0.38 + 0.07 * (angle * 47.0).sin();
            (
                width * (0.5 + radius * angle.cos()),
                height * (0.5 + radius * angle.sin()),
            )
        })
        .collect()
}

#[test]
fn polygon_scan_preserves_clipping_holes_alpha_and_invalid_edges() {
    let fixtures = [
        vec![
            detailed_ring(2048, 96.0, 80.0),
            detailed_ring(41, 48.0, 40.0),
        ],
        vec![vec![
            (-90.0, -80.0),
            (200.0, 0.5),
            (50.0, 200.0),
            (30.0, 0.5),
        ]],
        vec![vec![(0.5, 0.5), (80.5, 0.5), (80.5, 70.5), (0.5, 70.5)]],
        vec![vec![(2.0, 2.0), (90.0, 70.0), (2.0, 70.0), (90.0, 2.0)]],
        vec![vec![
            (f64::NAN, 1.0),
            (1.0, 2.0),
            (f64::INFINITY, 40.0),
            (70.0, 60.0),
        ]],
        vec![vec![(0.0, 2.0), (90.0, 2.0 + 1e-10), (40.0, 70.0)]],
        vec![vec![
            (-f64::MAX, 2.0),
            (f64::MAX, 60.0),
            (50.0, 60.0),
            (20.0, 2.0),
        ]],
        vec![],
        vec![vec![]],
        vec![vec![(4.0, 4.0)]],
    ];
    for rings in &fixtures {
        for clip in [
            None,
            Some((7, 9, 83, 66)),
            Some((-50, -30, 140, 100)),
            Some((9, 8, 3, 2)),
        ] {
            for alpha in [0, 97, 255] {
                let color = Rgba {
                    r: 30,
                    g: 80,
                    b: 170,
                    a: alpha,
                };
                let mut actual = RgbaImage::from_pixel(96, 80, image::Rgba([180, 190, 200, 255]));
                let mut expected = actual.clone();
                fill_polygon(&mut actual, rings, color, clip);
                reference_fill(&mut expected, rings, color, clip);
                assert_eq!(actual, expected, "clip={clip:?} alpha={alpha}");
            }
        }
    }
}

#[test]
#[ignore = "release-mode performance check on an otherwise idle compute worker"]
fn detailed_polygon_scan_avoids_inactive_edges() {
    use std::hint::black_box;
    use std::time::Instant;
    let rings = vec![detailed_ring(32_768, 480.0, 360.0)];
    let mut actual = RgbaImage::new(480, 360);
    let mut expected = actual.clone();
    let mut candidate = Vec::new();
    let mut reference = Vec::new();
    for _ in 0..9 {
        let started = Instant::now();
        fill_polygon(black_box(&mut actual), black_box(&rings), Rgba::BLACK, None);
        candidate.push(started.elapsed());
        let started = Instant::now();
        reference_fill(
            black_box(&mut expected),
            black_box(&rings),
            Rgba::BLACK,
            None,
        );
        reference.push(started.elapsed());
    }
    assert_eq!(actual, expected);
    candidate.sort();
    reference.sort();
    eprintln!("candidate={:?} reference={:?}", candidate[4], reference[4]);
    assert!(
        candidate[4] * 3 < reference[4],
        "inactive edges must not be scanned on every row"
    );
}
