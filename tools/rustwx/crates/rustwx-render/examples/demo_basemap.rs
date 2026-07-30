//! Render-only preview that exercises the full basemap stack the way production
//! maps should: ocean fill + land fill + coastlines/borders, with masked CAPE
//! data painted on top so the basemap shows through wherever data is zero.
//!
//! Local to the render crates — no product/planner code imported.
use rustwx_render::{
    BasemapStyle, Color, ColorScale, ContourLinePattern, ContourStyle, DiscreteColorScale,
    DomainFrame, ExtendMode, Field2D, GridShape, LambertConformal, LatLonGrid, LegendControls,
    LegendMode, MapRenderRequest, ProductKey, ProductVisualMode, ProjectedDomain, ProjectedExtent,
    ProjectedLineOverlay, ProjectedMapBuildOptions, ProjectedPolygonFill, ProjectionSpec,
    RenderDensity, StyledLonLatLayer, StyledLonLatPolygonLayer, WeatherPalette, WeatherProduct,
    WindStreamlineStyle, build_projected_map_with_options, load_styled_conus_features_for,
    load_styled_conus_polygons_for, palette_scale, save_png,
};
use std::path::PathBuf;

struct GalleryCase {
    filename: &'static str,
    title: &'static str,
    description: &'static str,
}

// CONUS-centered Lambert Conformal Conic — matches HRRR-ish defaults.
const TRUE_LAT_1: f64 = 33.0;
const TRUE_LAT_2: f64 = 45.0;
const STAND_LON: f64 = -98.0;
const REF_LAT: f64 = 38.5;

// CONUS-proper extent shifted far enough south to keep the Florida Keys in
// frame. The projected frame is sampled along every edge below; using only
// the four corners clips the bowed Lambert southern edge.
const FRAME_LAT_MIN: f64 = 21.5;
const FRAME_LAT_MAX: f64 = 50.0;
const FRAME_LON_MIN: f64 = -124.0;
const FRAME_LON_MAX: f64 = -66.0;
const FRAME_EDGE_SAMPLES: usize = 96;

const NX: usize = 540;
const NY: usize = 320;

fn main() {
    let proj = LambertConformal::new(TRUE_LAT_1, TRUE_LAT_2, STAND_LON, REF_LAT);

    let extent = projected_frame_extent(&proj);

    let shape = GridShape::new(NX, NY).expect("valid grid");
    let len = shape.len();
    let mut lat = Vec::with_capacity(len);
    let mut lon = Vec::with_capacity(len);
    let mut proj_x = Vec::with_capacity(len);
    let mut proj_y = Vec::with_capacity(len);
    let mut values = Vec::with_capacity(len);
    let mut height_values = Vec::with_capacity(len);
    let mut thickness_values = Vec::with_capacity(len);
    let mut u_values = Vec::with_capacity(len);
    let mut v_values = Vec::with_capacity(len);

    for j in 0..NY {
        let fy = j as f64 / (NY - 1) as f64;
        let y = extent.y_min + fy * (extent.y_max - extent.y_min);
        for i in 0..NX {
            let fx = i as f64 / (NX - 1) as f64;
            let x = extent.x_min + fx * (extent.x_max - extent.x_min);
            let (approx_lat, approx_lon) = inverse_lambert(&proj, x, y);

            // Synthetic CAPE-like field confined to the plains / southern US.
            // Values outside the blob stay below 250 so mask_below drops them
            // to fully transparent, letting the land fill show through.
            let dx = (approx_lon - -97.0) / 8.0;
            let dy = (approx_lat - 36.0) / 5.0;
            let blob = (-((dx * dx) + (dy * dy))).exp() * 4200.0;
            let tongue = (-(((approx_lon - -89.0) / 6.0).powi(2)
                + ((approx_lat - 32.5) / 3.0).powi(2)))
            .exp()
                * 3000.0;
            let cape = (blob + tongue).max(0.0);
            values.push(cape as f32);

            // Synthetic 500 mb height surrogate for contour overlay (m).
            let lat_comp = 5820.0 - ((approx_lat - 37.5).abs() * 8.0);
            let wave = ((approx_lon + 98.0) * 0.10).sin() * 35.0;
            let trough = (-(((approx_lon - -108.0) / 12.0).powi(2)
                + ((approx_lat - 41.0) / 7.0).powi(2)))
            .exp()
                * -90.0;
            let ridge = (-(((approx_lon - -82.0) / 12.0).powi(2)
                + ((approx_lat - 36.0) / 7.0).powi(2)))
            .exp()
                * 60.0;
            let height = lat_comp + wave + trough + ridge;
            height_values.push(height as f32);

            // Synthetic 1000-500 mb thickness surrogate (dam), used only to
            // exercise dashed synoptic overlay styling in the proof gallery.
            let thickness = 552.0 - (approx_lat - 39.0) * 1.6
                + ((approx_lon + 101.0) * 0.12).sin() * 6.0
                + trough * 0.04
                + ridge * 0.03;
            thickness_values.push(thickness as f32);

            let jet_core = (-(((approx_lon - -92.0) / 18.0).powi(2)
                + ((approx_lat - 41.0) / 5.5).powi(2)))
            .exp();
            let southerly_jet = (-(((approx_lon - -96.0) / 7.0).powi(2)
                + ((approx_lat - 33.0) / 5.0).powi(2)))
            .exp();
            let u_wind = 24.0 + jet_core * 52.0 + ((approx_lat - 36.0) * 1.2);
            let v_wind = 8.0 * ((approx_lon + 101.0) * 0.16).sin() + southerly_jet * 32.0
                - jet_core * ((approx_lon + 94.0) * 0.20).sin() * 18.0;
            u_values.push(u_wind as f32);
            v_values.push(v_wind as f32);

            lat.push(approx_lat as f32);
            lon.push(approx_lon as f32);
            proj_x.push(x);
            proj_y.push(y);
        }
    }

    let grid = LatLonGrid::new(shape, lat, lon).expect("grid");
    let field =
        Field2D::new(ProductKey::named("SBECAPE"), "J/kg", grid.clone(), values).expect("field");
    let height_field = Field2D::new(
        ProductKey::named("HEIGHT"),
        "m",
        grid.clone(),
        height_values,
    )
    .expect("height field");
    let thickness_field = Field2D::new(
        ProductKey::named("THICKNESS"),
        "dam",
        grid.clone(),
        thickness_values,
    )
    .expect("thickness field");
    let u_field =
        Field2D::new(ProductKey::named("U_WIND"), "kt", grid.clone(), u_values).expect("u field");
    let v_field =
        Field2D::new(ProductKey::named("V_WIND"), "kt", grid.clone(), v_values).expect("v field");

    let proof_dir = workspace_proof_dir();
    std::fs::create_dir_all(&proof_dir).expect("proof dir");
    let mut gallery_cases = Vec::new();

    for (style, filename, title, description) in [
        (
            BasemapStyle::Filled,
            "rustwx_render_demo_basemap.png",
            "Filled Basemap",
            "Masked CAPE over land/ocean fills with clean-atlas line hierarchy.",
        ),
        (
            BasemapStyle::White,
            "rustwx_render_demo_basemap_white.png",
            "White Basemap",
            "NWS-style white basemap variant for saturated diagnostic overlays.",
        ),
    ] {
        let mut request = MapRenderRequest::new(field.clone(), cape_scale_masked());
        request.title = Some(match style {
            BasemapStyle::Filled => "SBECAPE — filled basemap".to_string(),
            BasemapStyle::White => "SBECAPE — white basemap (NWS-style)".to_string(),
        });
        request.subtitle_left = Some("Synthetic field · Lambert Conformal CONUS".to_string());
        request.subtitle_right = Some("rustwx-render native engine".to_string());
        request.cbar_tick_step = Some(500.0);
        request.domain_frame = Some(DomainFrame::model_data_default());
        request.projected_domain = Some(ProjectedDomain {
            x: proj_x.clone(),
            y: proj_y.clone(),
            extent: extent.clone(),
        });
        request.projected_polygons = project_polygons(&proj, &extent, style);
        request.projected_lines = project_lines(&proj, &extent, style);
        request = request
            .with_contour_field(
                &height_field,
                (5640..=5880).step_by(30).map(|h| h as f64).collect(),
                ContourStyle {
                    color: Color::rgba(30, 34, 44, 200),
                    width: 1,
                    labels: true,
                    show_extrema: true,
                    major_every: Some(2),
                    major_width: Some(2),
                    ..Default::default()
                },
            )
            .expect("height contour")
            .with_contour_field(
                &thickness_field,
                (528..=582).step_by(6).map(|h| h as f64).collect(),
                ContourStyle {
                    color: Color::rgba(235, 30, 55, 215),
                    width: 1,
                    labels: true,
                    pattern: ContourLinePattern::Dashed,
                    major_every: Some(2),
                    major_width: Some(1),
                    ..Default::default()
                },
            )
            .expect("contour")
            .with_wind_streamlines(
                &u_field,
                &v_field,
                WindStreamlineStyle {
                    stride_x: 16,
                    stride_y: 14,
                    color: Color::rgba(18, 24, 32, 58),
                    width: 1,
                    max_steps: 16,
                    step_cells: 0.80,
                    min_speed: 5.0,
                },
            )
            .expect("streamlines");

        let output = proof_dir.join(filename);
        save_png(&request, &output).expect("render png");
        println!("{}", output.display());
        gallery_cases.push(GalleryCase {
            filename,
            title,
            description,
        });
    }

    let qpf_field = Field2D::new(
        ProductKey::named("TOTAL_QPF"),
        "in",
        grid.clone(),
        synthetic_qpf_values(&grid),
    )
    .expect("qpf field");
    let mut qpf_request = MapRenderRequest::new(qpf_field, precip_scale());
    qpf_request.title = Some("Total QPF - long-range texture case".to_string());
    qpf_request.subtitle_left = Some("Synthetic accumulated precipitation".to_string());
    qpf_request.subtitle_right = Some("Masked lows + clean-atlas linework".to_string());
    qpf_request.cbar_tick_step = Some(1.0);
    qpf_request.visual_mode = ProductVisualMode::FilledMeteorology;
    apply_conus_context(
        &mut qpf_request,
        &proj,
        &extent,
        &proj_x,
        &proj_y,
        BasemapStyle::Filled,
    );
    save_quality_case(
        &proof_dir,
        &mut gallery_cases,
        qpf_request,
        "rustwx_plot_quality_conus_qpf.png",
        "CONUS QPF",
        "Accumulated precipitation texture case for checking posterization and linework balance.",
    );

    let sparse_field = Field2D::new(
        ProductKey::named("SCP_PROXY"),
        "index",
        grid.clone(),
        synthetic_sparse_severe_values(&grid),
    )
    .expect("sparse severe field");
    let mut sparse_request =
        MapRenderRequest::for_weather_product(sparse_field, WeatherProduct::Scp);
    sparse_request.title = Some("SCP proxy - sparse field case".to_string());
    sparse_request.subtitle_left = Some("Synthetic sparse severe diagnostic".to_string());
    sparse_request.subtitle_right =
        Some("Near-zero values should not read as failed render".to_string());
    sparse_request.cbar_tick_step = Some(1.0);
    apply_conus_context(
        &mut sparse_request,
        &proj,
        &extent,
        &proj_x,
        &proj_y,
        BasemapStyle::Filled,
    );
    save_quality_case(
        &proof_dir,
        &mut gallery_cases,
        sparse_request,
        "rustwx_plot_quality_sparse_severe.png",
        "Sparse Severe Diagnostic",
        "Sparse diagnostic case for verifying empty/near-zero treatment and basemap restraint.",
    );

    let temperature_field = Field2D::new(
        ProductKey::named("LOCAL_TEMPERATURE"),
        "degF",
        grid.clone(),
        synthetic_local_temperature_values(&grid),
    )
    .expect("temperature field");
    let mut temperature_request = MapRenderRequest::new(temperature_field, temperature_scale());
    temperature_request.title = Some("2m temperature - local contrast case".to_string());
    temperature_request.subtitle_left = Some("Synthetic regional warm-sector field".to_string());
    temperature_request.subtitle_right = Some("Checks broad-scale palette saturation".to_string());
    temperature_request.cbar_tick_step = Some(5.0);
    temperature_request.visual_mode = ProductVisualMode::FilledMeteorology;
    temperature_request.legend = LegendControls {
        mode: LegendMode::SmoothRamp,
        ..LegendControls::default()
    };
    apply_conus_context(
        &mut temperature_request,
        &proj,
        &extent,
        &proj_x,
        &proj_y,
        BasemapStyle::White,
    );
    save_quality_case(
        &proof_dir,
        &mut gallery_cases,
        temperature_request,
        "rustwx_plot_quality_local_temperature.png",
        "Local Temperature",
        "Regional temperature case for checking whether the palette preserves useful local contrast.",
    );

    let reflectivity_field = Field2D::new(
        ProductKey::named("COMPOSITE_REFLECTIVITY"),
        "dBZ",
        grid.clone(),
        vec![0.0; grid.shape.len()],
    )
    .expect("reflectivity field");
    let mut reflectivity_request = MapRenderRequest::new(reflectivity_field, reflectivity_scale());
    reflectivity_request.title = Some("Composite reflectivity - no returns case".to_string());
    reflectivity_request.subtitle_left = Some("Synthetic empty radar-like field".to_string());
    reflectivity_request.subtitle_right =
        Some("Basemap-only output should still look intentional".to_string());
    reflectivity_request.cbar_tick_step = Some(10.0);
    reflectivity_request.visual_mode = ProductVisualMode::FilledMeteorology;
    apply_conus_context(
        &mut reflectivity_request,
        &proj,
        &extent,
        &proj_x,
        &proj_y,
        BasemapStyle::Filled,
    );
    save_quality_case(
        &proof_dir,
        &mut gallery_cases,
        reflectivity_request,
        "rustwx_plot_quality_empty_reflectivity.png",
        "Empty Reflectivity",
        "No-signal case for checking that masked products do not look like broken renders.",
    );

    let (global_grid, global_values) = synthetic_global_temperature_grid();
    let global_field = Field2D::new(
        ProductKey::named("GLOBAL_TEMPERATURE"),
        "degC",
        global_grid.clone(),
        global_values,
    )
    .expect("global field");
    let mut global_request = MapRenderRequest::new(global_field, global_temperature_scale());
    global_request.title = Some("Global temperature - broad basemap case".to_string());
    global_request.subtitle_left = Some("Synthetic global field".to_string());
    global_request.subtitle_right =
        Some("Robinson projection without default graticule clutter".to_string());
    global_request.width = 1400;
    global_request.height = 1000;
    global_request.visual_mode = ProductVisualMode::FilledMeteorology;
    global_request.render_density = RenderDensity::default();
    global_request.legend = LegendControls {
        mode: LegendMode::SmoothRamp,
        ..LegendControls::default()
    };
    let global_projected = build_projected_map_with_options(
        &global_grid.lat_deg,
        &global_grid.lon_deg,
        &ProjectedMapBuildOptions::from_bounds((-180.0, 179.999, -90.0, 90.0), 1.4)
            .with_projection(ProjectionSpec::Robinson {
                central_meridian_deg: 0.0,
            }),
    )
    .expect("global projected map");
    global_request.apply_projected_map(&global_projected);
    save_quality_case(
        &proof_dir,
        &mut gallery_cases,
        global_request,
        "rustwx_plot_quality_global_temperature.png",
        "Global Temperature",
        "Global filled-field case for checking basemap density, whitespace, and colorbar treatment.",
    );
    let (html, manifest) = write_gallery_index(&proof_dir, &gallery_cases).expect("gallery index");
    println!("{}", html.display());
    println!("{}", manifest.display());
}

/// SBECAPE colorscale with mask_below(250) so low-CAPE cells render transparent
/// and the underlying land fill shows through — matches the reference images'
/// behavior where "no snowfall" or "no precipitation" cells let the basemap show.
fn cape_scale_masked() -> ColorScale {
    use rustwx_render::weather::WeatherPreset;
    // Start from the Weather CAPE palette so we keep the editorial color ramp,
    // then override with mask_below.
    let base = WeatherPreset::Cape.scale();
    ColorScale::Discrete(DiscreteColorScale {
        levels: base.levels,
        colors: base.colors,
        extend: ExtendMode::Max,
        mask_below: Some(250.0),
    })
}

fn apply_conus_context(
    request: &mut MapRenderRequest,
    proj: &LambertConformal,
    extent: &ProjectedExtent,
    proj_x: &[f64],
    proj_y: &[f64],
    style: BasemapStyle,
) {
    request.width = 1400;
    request.height = 1000;
    request.domain_frame = Some(DomainFrame::map_viewport_default());
    request.projected_domain = Some(ProjectedDomain {
        x: proj_x.to_vec(),
        y: proj_y.to_vec(),
        extent: extent.clone(),
    });
    request.projected_polygons = project_polygons(proj, extent, style);
    request.projected_lines = project_lines(proj, extent, style);
}

fn save_quality_case(
    proof_dir: &std::path::Path,
    cases: &mut Vec<GalleryCase>,
    request: MapRenderRequest,
    filename: &'static str,
    title: &'static str,
    description: &'static str,
) {
    let output = proof_dir.join(filename);
    save_png(&request, &output).expect("render png");
    println!("{}", output.display());
    cases.push(GalleryCase {
        filename,
        title,
        description,
    });
}

fn precip_scale() -> ColorScale {
    ColorScale::Discrete(palette_scale(
        WeatherPalette::Precip,
        vec![
            0.0, 0.01, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
            0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
            1.8, 1.9, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0,
            8.0, 9.0, 10.0, 12.0, 15.0,
        ],
        ExtendMode::Max,
        Some(0.01),
    ))
}

fn reflectivity_scale() -> ColorScale {
    ColorScale::Discrete(palette_scale(
        WeatherPalette::Reflectivity,
        range_step(5.0, 80.0, 5.0),
        ExtendMode::Max,
        Some(5.0),
    ))
}

fn temperature_scale() -> ColorScale {
    ColorScale::Discrete(palette_scale(
        WeatherPalette::Temperature,
        range_step(-20.0, 116.0, 5.0),
        ExtendMode::Both,
        None,
    ))
}

fn global_temperature_scale() -> ColorScale {
    ColorScale::Discrete(palette_scale(
        WeatherPalette::Temperature,
        range_step(-60.0, 51.0, 5.0),
        ExtendMode::Both,
        None,
    ))
}

fn range_step(start: f64, stop_inclusive: f64, step: f64) -> Vec<f64> {
    let mut out = Vec::new();
    let mut value = start;
    while value <= stop_inclusive + step * 0.01 {
        out.push((value * 1000.0).round() / 1000.0);
        value += step;
    }
    out
}

fn synthetic_qpf_values(grid: &LatLonGrid) -> Vec<f32> {
    grid.lat_deg
        .iter()
        .zip(&grid.lon_deg)
        .map(|(&lat, &lon)| {
            let lat = lat as f64;
            let lon = lon as f64;
            let gulf_feed = gaussian(lon, lat, -91.0, 31.0, 5.5, 2.5) * 2.6;
            let plains_band = gaussian(lon, lat, -96.0, 37.0, 10.0, 2.8) * 1.8;
            let appalachian = gaussian(lon, lat, -83.0, 36.0, 3.5, 6.0) * 1.1;
            let embedded = (((lon + 98.0) * 0.9).sin() * ((lat - 30.0) * 1.7).cos()).max(0.0);
            let dry_slot = gaussian(lon, lat, -101.0, 34.0, 5.0, 4.0) * 1.5;
            let envelope = (gulf_feed + plains_band + appalachian).max(0.0);
            let texture = embedded * 0.24 * (envelope / 1.2).clamp(0.0, 1.0);
            let value = (envelope + texture - dry_slot).max(0.0).powf(1.15);
            if value < 0.01 { 0.0 } else { value as f32 }
        })
        .collect()
}

fn synthetic_sparse_severe_values(grid: &LatLonGrid) -> Vec<f32> {
    grid.lat_deg
        .iter()
        .zip(&grid.lon_deg)
        .map(|(&lat, &lon)| {
            let lat = lat as f64;
            let lon = lon as f64;
            let corridor = gaussian(lon, lat, -97.5, 36.0, 3.5, 1.8) * 8.0;
            let secondary = gaussian(lon, lat, -89.5, 33.0, 2.5, 1.6) * 3.0;
            let wave = (((lon + 100.0) * 2.3).sin() + ((lat - 34.0) * 2.7).cos()).max(0.0);
            let value = corridor * (0.62 + wave * 0.18) + secondary;
            if value < 1.0 { 0.0 } else { value as f32 }
        })
        .collect()
}

fn synthetic_local_temperature_values(grid: &LatLonGrid) -> Vec<f32> {
    grid.lat_deg
        .iter()
        .zip(&grid.lon_deg)
        .map(|(&lat, &lon)| {
            let lat = lat as f64;
            let lon = lon as f64;
            let south_to_north = 96.0 - (lat - 27.0) * 1.15;
            let dryline = 6.0 / (1.0 + (-(lon + 99.0) * 1.5).exp());
            let urban_heat = gaussian(lon, lat, -97.5, 35.5, 1.7, 1.2) * 4.0;
            let cloud_shadow = gaussian(lon, lat, -92.0, 36.5, 4.0, 2.0) * -5.5;
            (south_to_north + dryline + urban_heat + cloud_shadow) as f32
        })
        .collect()
}

fn synthetic_global_temperature_grid() -> (LatLonGrid, Vec<f32>) {
    let nx = 361;
    let ny = 181;
    let shape = GridShape::new(nx, ny).expect("global grid shape");
    let mut lat_deg = Vec::with_capacity(shape.len());
    let mut lon_deg = Vec::with_capacity(shape.len());
    let mut values = Vec::with_capacity(shape.len());

    for j in 0..ny {
        let lat = -90.0 + j as f64;
        for i in 0..nx {
            let lon = -180.0 + i as f64;
            let seasonal_wave = ((lon + 35.0).to_radians() * 2.0).sin() * 5.0;
            let lat_gradient = 31.0 - lat.abs() * 0.86;
            let land_proxy = ((lon * 0.08).sin() * (lat * 0.13).cos()).max(0.0) * 8.0;
            let polar_cold = gaussian(lon, lat, 20.0, -73.0, 45.0, 10.0) * -20.0;
            let value = lat_gradient + seasonal_wave + land_proxy + polar_cold;
            lat_deg.push(lat as f32);
            lon_deg.push(lon as f32);
            values.push(value as f32);
        }
    }

    (
        LatLonGrid::new(shape, lat_deg, lon_deg).expect("global grid"),
        values,
    )
}

fn gaussian(lon: f64, lat: f64, lon0: f64, lat0: f64, sx: f64, sy: f64) -> f64 {
    let dx = (lon - lon0) / sx;
    let dy = (lat - lat0) / sy;
    (-(dx * dx + dy * dy)).exp()
}

fn projected_frame_extent(proj: &LambertConformal) -> ProjectedExtent {
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    let mut include = |lat: f64, lon: f64| {
        let (x, y) = proj.project(lat, lon);
        if x.is_finite() && y.is_finite() {
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }
    };

    for n in 0..=FRAME_EDGE_SAMPLES {
        let t = n as f64 / FRAME_EDGE_SAMPLES as f64;
        let lon = FRAME_LON_MIN + t * (FRAME_LON_MAX - FRAME_LON_MIN);
        let lat = FRAME_LAT_MIN + t * (FRAME_LAT_MAX - FRAME_LAT_MIN);
        include(FRAME_LAT_MIN, lon);
        include(FRAME_LAT_MAX, lon);
        include(lat, FRAME_LON_MIN);
        include(lat, FRAME_LON_MAX);
    }

    ProjectedExtent {
        x_min,
        x_max,
        y_min,
        y_max,
    }
}

fn project_polygons(
    proj: &LambertConformal,
    extent: &ProjectedExtent,
    style: BasemapStyle,
) -> Vec<ProjectedPolygonFill> {
    let layers: Vec<StyledLonLatPolygonLayer> = load_styled_conus_polygons_for(style);
    let mut out = Vec::new();

    // Pad the accept window generously — polygons extend beyond the frame and
    // the scanline fill clips to image bounds anyway.
    let pad_x = 0.50 * (extent.x_max - extent.x_min);
    let pad_y = 0.50 * (extent.y_max - extent.y_min);
    let bbox = (
        extent.x_min - pad_x,
        extent.x_max + pad_x,
        extent.y_min - pad_y,
        extent.y_max + pad_y,
    );

    for layer in layers {
        let color = Color::rgba(layer.color.r, layer.color.g, layer.color.b, layer.color.a);
        for polygon in layer.polygons {
            let rings: Vec<Vec<(f64, f64)>> = polygon
                .into_iter()
                .map(|ring| {
                    ring.into_iter()
                        .map(|(lon, lat)| proj.project(lat, lon))
                        .collect::<Vec<(f64, f64)>>()
                })
                .filter(|ring| ring_overlaps_bbox(ring, bbox))
                .collect();
            if rings.is_empty() {
                continue;
            }
            out.push(ProjectedPolygonFill {
                rings,
                color,
                role: layer.role,
            });
        }
    }
    out
}

fn ring_overlaps_bbox(ring: &[(f64, f64)], bbox: (f64, f64, f64, f64)) -> bool {
    let (mut rx_min, mut rx_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut ry_min, mut ry_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(x, y) in ring {
        if x < rx_min {
            rx_min = x;
        }
        if x > rx_max {
            rx_max = x;
        }
        if y < ry_min {
            ry_min = y;
        }
        if y > ry_max {
            ry_max = y;
        }
    }
    !(rx_max < bbox.0 || rx_min > bbox.1 || ry_max < bbox.2 || ry_min > bbox.3)
}

fn project_lines(
    proj: &LambertConformal,
    extent: &ProjectedExtent,
    style: BasemapStyle,
) -> Vec<ProjectedLineOverlay> {
    let layers: Vec<StyledLonLatLayer> = load_styled_conus_features_for(style);
    let mut overlays = Vec::new();
    let pad_x = 0.10 * (extent.x_max - extent.x_min);
    let pad_y = 0.10 * (extent.y_max - extent.y_min);
    let x_lo = extent.x_min - pad_x;
    let x_hi = extent.x_max + pad_x;
    let y_lo = extent.y_min - pad_y;
    let y_hi = extent.y_max + pad_y;

    for layer in layers {
        let color = Color::rgba(layer.color.r, layer.color.g, layer.color.b, layer.color.a);
        for line in layer.lines {
            let mut current: Vec<(f64, f64)> = Vec::with_capacity(line.len());
            for (lon, lat) in line {
                let (x, y) = proj.project(lat, lon);
                if x < x_lo || x > x_hi || y < y_lo || y > y_hi {
                    if current.len() >= 2 {
                        overlays.push(ProjectedLineOverlay {
                            points: std::mem::take(&mut current),
                            color,
                            width: layer.width,
                            role: layer.role,
                        });
                    } else {
                        current.clear();
                    }
                    continue;
                }
                current.push((x, y));
            }
            if current.len() >= 2 {
                overlays.push(ProjectedLineOverlay {
                    points: current,
                    color,
                    width: layer.width,
                    role: layer.role,
                });
            }
        }
    }
    overlays
}

/// Newton iteration back from projected (x, y) to (lat, lon) — synthetic-only.
fn inverse_lambert(proj: &LambertConformal, x: f64, y: f64) -> (f64, f64) {
    let mut lat = REF_LAT;
    let mut lon = STAND_LON + (x / 100_000.0);
    for _ in 0..40 {
        let (px, py) = proj.project(lat, lon);
        let ex = x - px;
        let ey = y - py;
        if ex.abs() < 10.0 && ey.abs() < 10.0 {
            break;
        }
        let eps = 1e-3;
        let (px1, py1) = proj.project(lat + eps, lon);
        let (px2, py2) = proj.project(lat, lon + eps);
        let j11 = (px1 - px) / eps;
        let j12 = (px2 - px) / eps;
        let j21 = (py1 - py) / eps;
        let j22 = (py2 - py) / eps;
        let det = j11 * j22 - j12 * j21;
        if det.abs() < 1e-12 {
            break;
        }
        let dlat = (j22 * ex - j12 * ey) / det;
        let dlon = (-j21 * ex + j11 * ey) / det;
        lat += dlat.clamp(-3.0, 3.0);
        lon += dlon.clamp(-3.0, 3.0);
    }
    (lat, lon)
}

fn workspace_proof_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("proof")
}

fn write_gallery_index(
    proof_dir: &std::path::Path,
    cases: &[GalleryCase],
) -> std::io::Result<(PathBuf, PathBuf)> {
    let gallery_dir = proof_dir.join("plot_quality_gallery");
    std::fs::create_dir_all(&gallery_dir)?;
    let manifest_path = gallery_dir.join("index.json");
    let html_path = gallery_dir.join("index.html");

    let mut manifest =
        String::from("{\n  \"title\": \"RustWX Plot Quality Gallery\",\n  \"cases\": [\n");
    for (idx, case) in cases.iter().enumerate() {
        let comma = if idx + 1 == cases.len() { "" } else { "," };
        manifest.push_str(&format!(
            "    {{ \"filename\": \"{}\", \"title\": \"{}\", \"description\": \"{}\" }}{}\n",
            json_escape(case.filename),
            json_escape(case.title),
            json_escape(case.description),
            comma
        ));
    }
    manifest.push_str("  ]\n}\n");
    std::fs::write(&manifest_path, manifest)?;

    let mut html = String::from(
        "<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>RustWX Plot Quality Gallery</title><style>\n",
    );
    html.push_str(
        "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;background:#f5f7f9;color:#17202a}main{max-width:1180px;margin:0 auto}section{margin:24px 0}img{display:block;max-width:100%;height:auto;border:1px solid #c9d1d9;background:white}h1{font-size:24px}h2{font-size:18px;margin-bottom:6px}p{margin-top:0;color:#526171}\n",
    );
    html.push_str("</style></head><body><main><h1>RustWX Plot Quality Gallery</h1>\n");
    html.push_str("<p>Stable render-only cases for reviewing basemap hierarchy, masked weather fills, chrome, and colorbar treatment.</p>\n");
    for case in cases {
        html.push_str("<section>");
        html.push_str(&format!(
            "<h2>{}</h2><p>{}</p><img src=\"../{}\" alt=\"{}\">",
            html_escape(case.title),
            html_escape(case.description),
            html_escape(case.filename),
            html_escape(case.title)
        ));
        html.push_str("</section>\n");
    }
    html.push_str("</main></body></html>\n");
    std::fs::write(&html_path, html)?;

    Ok((html_path, manifest_path))
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
