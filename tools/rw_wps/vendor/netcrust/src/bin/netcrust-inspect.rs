use std::path::PathBuf;

fn main() -> netcrust::Result<()> {
    let Some(path) = std::env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: netcrust-inspect <file> [variable]");
        std::process::exit(2);
    };

    let file = netcrust::open(&path)?;
    println!("path: {}", path.display());
    println!("format: {:?}", file.format());

    println!("dimensions:");
    for dim in file.dimensions()? {
        println!(
            "  {} = {}{}",
            dim.name(),
            dim.len(),
            if dim.is_unlimited() {
                " (unlimited)"
            } else {
                ""
            }
        );
    }

    println!("attributes:");
    for attr in file.attributes()?.into_iter().take(24) {
        println!("  {} = {:?}", attr.name(), attr.value());
    }

    println!("variables:");
    for var in file.variables()?.into_iter().take(80) {
        println!("  {} {:?} {:?}", var.name(), var.dtype(), var.shape());
    }

    if let Some(variable) = std::env::args().nth(2) {
        let data = file.read_array_f64_first_record_or_all(&variable)?;
        println!(
            "read {} shape={:?} len={} first={:?}",
            variable,
            data.shape(),
            data.len(),
            data.values().first()
        );
    }

    Ok(())
}
