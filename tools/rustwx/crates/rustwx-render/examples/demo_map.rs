use rustwx_render::{
    Field2D, GridShape, LatLonGrid, MapRenderRequest, ProductKey, ProjectedDomain, ProjectedExtent,
    save_png,
    weather::{ECAPE_SEVERE_PANEL_PRODUCTS, WeatherProduct},
};
use std::path::PathBuf;

fn main() {
    let shape = GridShape::new(120, 80).expect("valid grid");
    let len = shape.len();
    let mut lat = Vec::with_capacity(len);
    let mut lon = Vec::with_capacity(len);
    let mut values = Vec::with_capacity(len);

    for j in 0..shape.ny {
        let y = j as f32 / (shape.ny - 1) as f32;
        let lat_value = 30.0 + y * 15.0;
        for i in 0..shape.nx {
            let x = i as f32 / (shape.nx - 1) as f32;
            let lon_value = -105.0 + x * 20.0;
            let dx = x - 0.58;
            let dy = y - 0.52;
            let blob = (-((dx * dx) / 0.018 + (dy * dy) / 0.02)).exp() * 3500.0;
            let ridge = ((x * 8.0).sin() * 0.5 + 0.5) * 500.0;
            lat.push(lat_value);
            lon.push(lon_value);
            values.push((blob + ridge) as f32);
        }
    }

    let grid = LatLonGrid::new(shape, lat, lon).expect("grid");
    let field = Field2D::new(ProductKey::named("SBECAPE"), "J/kg", grid, values).expect("field");
    let mut request = MapRenderRequest::for_weather_product(field, WeatherProduct::Sbecape);
    request.title = Some("rustwx Demo SBECAPE".to_string());
    request.subtitle_left = Some("Synthetic field".to_string());
    request.subtitle_right = Some(format!(
        "Weather palette | panel={}",
        ECAPE_SEVERE_PANEL_PRODUCTS.len()
    ));
    request.projected_domain = Some(ProjectedDomain {
        x: (0..shape.ny)
            .flat_map(|_| (0..shape.nx).map(|i| i as f64))
            .collect(),
        y: (0..shape.ny)
            .flat_map(|j| std::iter::repeat_n(j as f64, shape.nx))
            .collect(),
        extent: ProjectedExtent {
            x_min: 0.0,
            x_max: (shape.nx - 1) as f64,
            y_min: 0.0,
            y_max: (shape.ny - 1) as f64,
        },
    });

    let proof_dir = workspace_proof_dir();
    std::fs::create_dir_all(&proof_dir).expect("proof dir");
    let output = proof_dir.join("rustwx_render_demo_sbecape.png");
    save_png(&request, &output).expect("render png");
    println!("{}", output.display());
}

fn workspace_proof_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("proof")
}
