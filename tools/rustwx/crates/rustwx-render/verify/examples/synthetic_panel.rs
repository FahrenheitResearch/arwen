use rustwx_render_verify::{
    Color, PanelGridLayout, default_output_dir, render_panel_grid, sample_panel_requests,
    save_rgba_png_profile_with_options,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let requests = sample_panel_requests()?;
    let layout =
        PanelGridLayout::two_by_two(320, 220)?.with_background(Color::rgba(244, 244, 244, 255));
    let image = render_panel_grid(&layout, &requests)?;

    let output_dir = default_output_dir();
    std::fs::create_dir_all(&output_dir)?;
    let output_path = output_dir.join("synthetic_weather_native_panel.png");
    save_rgba_png_profile_with_options(&image, &output_path, &Default::default())?;
    println!("{}", output_path.display());
    Ok(())
}
