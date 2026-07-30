pub use rustwx_render::*;

use std::path::PathBuf;

const DEFAULT_WIDTH: u32 = 320;
const DEFAULT_HEIGHT: u32 = 220;

fn normalized(index: usize, max: usize) -> f32 {
    if max <= 1 {
        0.0
    } else {
        index as f32 / (max - 1) as f32
    }
}

pub fn synthetic_lat_lon_grid(nx: usize, ny: usize) -> Result<LatLonGrid, RustwxRenderError> {
    let shape = GridShape::new(nx, ny)?;
    let mut lat_deg = Vec::with_capacity(shape.len());
    let mut lon_deg = Vec::with_capacity(shape.len());

    for j in 0..shape.ny {
        let y = normalized(j, shape.ny);
        let lat = 28.0 + y * 18.0;
        for i in 0..shape.nx {
            let x = normalized(i, shape.nx);
            let lon = -109.0 + x * 24.0;
            lat_deg.push(lat);
            lon_deg.push(lon);
        }
    }

    LatLonGrid::new(shape, lat_deg, lon_deg)
}

pub fn synthetic_projected_domain(shape: GridShape) -> ProjectedDomain {
    ProjectedDomain {
        x: (0..shape.ny)
            .flat_map(|_| (0..shape.nx).map(|i| i as f64))
            .collect(),
        y: (0..shape.ny)
            .flat_map(|j| std::iter::repeat_n(j as f64, shape.nx))
            .collect(),
        extent: ProjectedExtent {
            x_min: 0.0,
            x_max: (shape.nx.saturating_sub(1)) as f64,
            y_min: 0.0,
            y_max: (shape.ny.saturating_sub(1)) as f64,
        },
    }
}

pub fn synthetic_slanted_projected_domain(shape: GridShape) -> ProjectedDomain {
    let mut x = Vec::with_capacity(shape.len());
    let mut y = Vec::with_capacity(shape.len());
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for j in 0..shape.ny {
        let y_norm = normalized(j, shape.ny) as f64;
        for i in 0..shape.nx {
            let x_norm = normalized(i, shape.nx) as f64;
            let px = i as f64 + y_norm * 10.0;
            let py = j as f64 + (x_norm - 0.5) * 4.0;
            x_min = x_min.min(px);
            x_max = x_max.max(px);
            y_min = y_min.min(py);
            y_max = y_max.max(py);
            x.push(px);
            y.push(py);
        }
    }

    ProjectedDomain {
        x,
        y,
        extent: ProjectedExtent {
            x_min,
            x_max,
            y_min,
            y_max,
        },
    }
}

pub fn synthetic_blob_field(
    product: &str,
    units: &str,
    nx: usize,
    ny: usize,
) -> Result<Field2D, RustwxRenderError> {
    let grid = synthetic_lat_lon_grid(nx, ny)?;
    let mut values = Vec::with_capacity(grid.shape.len());

    for j in 0..grid.shape.ny {
        let y = normalized(j, grid.shape.ny);
        for i in 0..grid.shape.nx {
            let x = normalized(i, grid.shape.nx);
            let dx = x - 0.58;
            let dy = y - 0.50;
            let blob = (-((dx * dx) / 0.02 + (dy * dy) / 0.03)).exp() * 3200.0;
            let ridge = ((x * 8.0).sin() * 0.5 + 0.5) * 450.0;
            let south_gradient = (1.0 - y) * 250.0;
            values.push(blob + ridge + south_gradient);
        }
    }

    Field2D::new(ProductKey::named(product), units, grid, values)
}

pub fn synthetic_contour_band_field(
    product: &str,
    units: &str,
    nx: usize,
    ny: usize,
) -> Result<Field2D, RustwxRenderError> {
    let grid = synthetic_lat_lon_grid(nx, ny)?;
    let mut values = Vec::with_capacity(grid.shape.len());

    for j in 0..grid.shape.ny {
        let _y = normalized(j, grid.shape.ny);
        for i in 0..grid.shape.nx {
            let x = normalized(i, grid.shape.nx);
            values.push(x * 40.0);
        }
    }

    Field2D::new(ProductKey::named(product), units, grid, values)
}

pub fn synthetic_signed_field(
    product: &str,
    units: &str,
    nx: usize,
    ny: usize,
) -> Result<Field2D, RustwxRenderError> {
    let grid = synthetic_lat_lon_grid(nx, ny)?;
    let mut values = Vec::with_capacity(grid.shape.len());

    for j in 0..grid.shape.ny {
        let y = normalized(j, grid.shape.ny);
        for i in 0..grid.shape.nx {
            let x = normalized(i, grid.shape.nx);
            let wave = ((x * std::f32::consts::TAU * 1.2).sin() * 7.5)
                + ((y * std::f32::consts::PI).cos() * 4.5);
            values.push(wave);
        }
    }

    Field2D::new(ProductKey::named(product), units, grid, values)
}

pub fn synthetic_height_field(nx: usize, ny: usize) -> Result<Field2D, RustwxRenderError> {
    let grid = synthetic_lat_lon_grid(nx, ny)?;
    let mut values = Vec::with_capacity(grid.shape.len());

    for j in 0..grid.shape.ny {
        let y = normalized(j, grid.shape.ny);
        for i in 0..grid.shape.nx {
            let x = normalized(i, grid.shape.nx);
            let base = 5400.0 + y * 420.0;
            let wave = (x * std::f32::consts::TAU * 2.0).sin() * 70.0;
            values.push(base + wave);
        }
    }

    Field2D::new(ProductKey::named("height"), "m", grid, values)
}

pub fn synthetic_wind_components(
    nx: usize,
    ny: usize,
) -> Result<(Field2D, Field2D), RustwxRenderError> {
    let u_grid = synthetic_lat_lon_grid(nx, ny)?;
    let v_grid = synthetic_lat_lon_grid(nx, ny)?;
    let mut u_values = Vec::with_capacity(u_grid.shape.len());
    let mut v_values = Vec::with_capacity(v_grid.shape.len());

    for j in 0..u_grid.shape.ny {
        let y = normalized(j, u_grid.shape.ny);
        for i in 0..u_grid.shape.nx {
            let x = normalized(i, u_grid.shape.nx);
            u_values.push(18.0 + x * 20.0);
            v_values.push(-6.0 + y * 18.0 + (x * std::f32::consts::PI).sin() * 2.0);
        }
    }

    let u = Field2D::new(ProductKey::named("u_wind"), "kt", u_grid, u_values)?;
    let v = Field2D::new(ProductKey::named("v_wind"), "kt", v_grid, v_values)?;
    Ok((u, v))
}

pub fn sample_weather_request(
    product: WeatherProduct,
) -> Result<MapRenderRequest, RustwxRenderError> {
    let field = synthetic_blob_field(product.slug(), "J/kg", 48, 32)?;
    let mut request = MapRenderRequest::for_weather_product(field, product);
    request.width = DEFAULT_WIDTH;
    request.height = DEFAULT_HEIGHT;
    request.projected_domain = Some(synthetic_projected_domain(request.field.grid.shape));
    request.subtitle_left = Some("Synthetic verification field".into());
    request.subtitle_right = Some("rustwx-render-verify".into());
    Ok(request)
}

pub fn sample_derived_request() -> Result<MapRenderRequest, RustwxRenderError> {
    let field = synthetic_signed_field("temperature_advection_850mb", "K/hr", 48, 32)?;
    let mut request = MapRenderRequest::for_derived_product(
        field,
        DerivedProductStyle::TemperatureAdvection850mb,
    );
    request.width = DEFAULT_WIDTH;
    request.height = DEFAULT_HEIGHT;
    request.projected_domain = Some(synthetic_projected_domain(request.field.grid.shape));
    request.subtitle_left = Some("Signed synthetic verification field".into());
    request.subtitle_right = Some("rustwx-render-verify".into());
    Ok(request)
}

pub fn sample_overlay_request() -> Result<MapRenderRequest, RustwxRenderError> {
    let height = synthetic_height_field(48, 32)?;
    let contours = height.clone();
    let (u_wind, v_wind) = synthetic_wind_components(48, 32)?;

    let mut request = MapRenderRequest::contour_only(height)
        .with_contour_field(
            &contours,
            vec![
                5400.0, 5460.0, 5520.0, 5580.0, 5640.0, 5700.0, 5760.0, 5820.0,
            ],
            ContourStyle {
                labels: true,
                show_extrema: true,
                ..Default::default()
            },
        )?
        .with_wind_barbs(
            &u_wind,
            &v_wind,
            WindBarbStyle {
                stride_x: 4,
                stride_y: 4,
                length_px: 16.0,
                ..Default::default()
            },
        )?;
    request.width = DEFAULT_WIDTH;
    request.height = DEFAULT_HEIGHT;
    request.title = Some("500 MB HEIGHT / WIND VERIFY".into());
    request.subtitle_left = Some("Contour-only overlay path".into());
    request.subtitle_right = Some("rustwx-render-verify".into());
    request.projected_domain = Some(synthetic_projected_domain(request.field.grid.shape));
    Ok(request)
}

fn contour_alignment_scale() -> ColorScale {
    ColorScale::Discrete(DiscreteColorScale {
        levels: vec![0.0, 10.0, 20.0, 30.0, 40.0],
        colors: vec![
            Color::rgba(45, 94, 179, 255),
            Color::rgba(53, 168, 107, 255),
            Color::rgba(240, 184, 62, 255),
            Color::rgba(216, 91, 60, 255),
        ],
        extend: ExtendMode::Neither,
        mask_below: None,
    })
}

pub fn sample_contour_fill_alignment_request() -> Result<MapRenderRequest, RustwxRenderError> {
    let fill = synthetic_contour_band_field("contour_fill_alignment", "unitless", 56, 36)?;
    let contour_field = fill.clone();
    let mut request = MapRenderRequest::new(fill, contour_alignment_scale()).with_contour_field(
        &contour_field,
        vec![10.0, 20.0, 30.0],
        ContourStyle {
            color: Color::BLACK,
            width: 1,
            labels: false,
            show_extrema: false,
            ..Default::default()
        },
    )?;
    request.width = 360;
    request.height = 240;
    request.colorbar = false;
    request.title = None;
    request.subtitle_left = None;
    request.subtitle_center = None;
    request.subtitle_right = None;
    request.projected_domain = Some(synthetic_projected_domain(request.field.grid.shape));
    Ok(request)
}

pub fn sample_projected_contour_request() -> Result<MapRenderRequest, RustwxRenderError> {
    let fill = synthetic_contour_band_field("projected_contours", "unitless", 56, 36)?;
    let contour_field = fill.clone();
    let mut request = MapRenderRequest::new(fill, contour_alignment_scale()).with_contour_field(
        &contour_field,
        vec![10.0, 20.0, 30.0],
        ContourStyle {
            color: Color::BLACK,
            width: 1,
            labels: false,
            show_extrema: false,
            ..Default::default()
        },
    )?;
    request.width = 360;
    request.height = 240;
    request.colorbar = false;
    request.title = None;
    request.subtitle_left = None;
    request.subtitle_center = None;
    request.subtitle_right = None;
    request.projected_domain = Some(synthetic_slanted_projected_domain(request.field.grid.shape));
    Ok(request)
}

pub fn sample_panel_requests() -> Result<Vec<MapRenderRequest>, RustwxRenderError> {
    Ok(vec![
        sample_weather_request(WeatherProduct::Sbecape)?,
        sample_weather_request(WeatherProduct::Mlecape)?,
        sample_derived_request()?,
        sample_overlay_request()?,
    ])
}

pub fn count_non_background_pixels(image: &RgbaImage, background: [u8; 4]) -> usize {
    image.pixels().filter(|pixel| pixel.0 != background).count()
}

pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

pub fn default_output_dir() -> PathBuf {
    workspace_root().join("target").join("rustwx-render-verify")
}
