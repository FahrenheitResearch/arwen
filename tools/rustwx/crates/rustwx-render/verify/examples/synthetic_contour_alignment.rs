use rustwx_render_verify::{
    default_output_dir, sample_contour_fill_alignment_request, sample_projected_contour_request,
    save_png,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = default_output_dir();
    std::fs::create_dir_all(&output_dir)?;

    let regular_path = output_dir.join("synthetic_contour_fill_alignment.png");
    let projected_path = output_dir.join("synthetic_projected_contour_alignment.png");

    save_png(&sample_contour_fill_alignment_request()?, &regular_path)?;
    save_png(&sample_projected_contour_request()?, &projected_path)?;

    println!("{}", regular_path.display());
    println!("{}", projected_path.display());
    Ok(())
}
