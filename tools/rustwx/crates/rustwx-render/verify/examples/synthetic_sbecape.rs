use rustwx_render_verify::{WeatherProduct, default_output_dir, sample_weather_request, save_png};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request = sample_weather_request(WeatherProduct::Sbecape)?;
    let output_dir = default_output_dir();
    std::fs::create_dir_all(&output_dir)?;
    let output_path = output_dir.join("synthetic_sbecape_verify.png");
    save_png(&request, &output_path)?;
    println!("{}", output_path.display());
    Ok(())
}
