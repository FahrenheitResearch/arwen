//! Convert MPAS history frames into wrfout-shaped frames the renderer takes.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rw_mpas::convert::{convert_frame, ConvertOptions};
use rw_mpas::history::{assert_mesh_agrees, read_history, Timestamp};
use rw_mpas::weights::{build_weights, read_mesh_coordinates};
use rw_mpas::window::Window;
use rw_store::netcdf_classic::NcFormat;

const USAGE: &str = "\
usage: rw_mpas_convert --history FILE... --mesh FILE --out-dir DIR
                       [--init FILE] [--mesh-sha256 HEX] [--init-sha256 HEX]
                       [--window focus|global] [--field-set full|surface]
                       [--simulation-start YYYY-MM-DD_HH:MM:SS]
                       [--prefix NAME] [--format cdf2|cdf5] [--clobber]
                       [--json FILE]
       rw_mpas_convert --help | --abi";

const ABI: &str = "gpuwm-rw-mpas-convert-v1\tCONVERTED\tWINDOW\tWEIGHTS\tFINISHED";

struct Args {
    history: Vec<PathBuf>,
    mesh: Option<PathBuf>,
    mesh_sha256: Option<String>,
    init: Option<PathBuf>,
    init_sha256: Option<String>,
    out_dir: Option<PathBuf>,
    window: String,
    field_set: String,
    simulation_start: Option<String>,
    prefix: String,
    format: NcFormat,
    clobber: bool,
    json: Option<PathBuf>,
}

fn parse() -> Result<Args, String> {
    let mut args = Args {
        history: Vec::new(),
        mesh: None,
        mesh_sha256: None,
        init: None,
        init_sha256: None,
        out_dir: None,
        window: "focus".to_string(),
        field_set: "full".to_string(),
        simulation_start: None,
        prefix: "wrfout_d01".to_string(),
        format: NcFormat::Offset64,
        clobber: false,
        json: None,
    };
    let mut raw = std::env::args().skip(1).peekable();
    while let Some(flag) = raw.next() {
        let mut value = || -> Result<String, String> {
            raw.next()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--history" => {
                while let Some(next) = raw.peek() {
                    if next.starts_with("--") {
                        break;
                    }
                    args.history.push(PathBuf::from(raw.next().unwrap()));
                }
                if args.history.is_empty() {
                    return Err("--history needs at least one file".to_string());
                }
            }
            "--mesh" => args.mesh = Some(PathBuf::from(value()?)),
            "--mesh-sha256" => args.mesh_sha256 = Some(value()?),
            "--init" => args.init = Some(PathBuf::from(value()?)),
            "--init-sha256" => args.init_sha256 = Some(value()?),
            "--out-dir" => args.out_dir = Some(PathBuf::from(value()?)),
            "--window" => args.window = value()?,
            "--field-set" => args.field_set = value()?,
            "--simulation-start" => args.simulation_start = Some(value()?),
            "--prefix" => args.prefix = value()?,
            "--format" => {
                args.format = match value()?.as_str() {
                    "cdf2" => NcFormat::Offset64,
                    "cdf5" => NcFormat::Data64,
                    other => return Err(format!("--format '{other}' is not cdf2 or cdf5")),
                }
            }
            "--clobber" => args.clobber = true,
            "--json" => args.json = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(args)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn run() -> Result<(), String> {
    let args = parse()?;
    let mesh_path = args.mesh.ok_or("--mesh is required")?;
    let out_dir = args.out_dir.ok_or("--out-dir is required")?;
    if args.history.is_empty() {
        return Err("--history is required".to_string());
    }
    let window = Window::named(&args.window).map_err(|e| e.to_string())?;
    let simulation_start = match &args.simulation_start {
        Some(text) => Some(Timestamp::parse(text).map_err(|e| e.to_string())?),
        None => None,
    };

    let started = std::time::Instant::now();
    let mesh = read_mesh_coordinates(&mesh_path, args.mesh_sha256.as_deref())
        .map_err(|e| e.to_string())?;
    let mesh_seconds = started.elapsed().as_secs_f64();
    let weights =
        build_weights(&mesh, &args.window, &window).map_err(|e| e.to_string())?;
    println!(
        "WINDOW\t{}\t{}x{}\tcells={}",
        args.window,
        weights.ny,
        weights.nx,
        weights.n_cells
    );
    println!(
        "WEIGHTS\t{}\tbuild_s={:.3}\tmesh_read_s={:.3}\tmax_nn_km={:.3}\tmean_nn_km={:.3}",
        weights.weights_sha256,
        weights.build_seconds,
        mesh_seconds,
        weights.max_distance_km(),
        weights.mean_distance_km()
    );

    let mut records: Vec<String> = Vec::new();
    for history in &args.history {
        let frame = read_history(
            history,
            args.init.as_deref(),
            None,
            None,
            args.init_sha256.as_deref(),
        )
        .map_err(|e| e.to_string())?;
        let (dlat, dlon) =
            assert_mesh_agrees(&frame, &mesh, 1.0e-4).map_err(|e| e.to_string())?;
        let stamp = frame.valid_time.label().replace(':', "_");
        let destination = Path::new(&out_dir).join(format!("{}_{stamp}", args.prefix));
        let options = ConvertOptions {
            simulation_start,
            field_set: args.field_set.clone(),
            format: args.format,
            clobber: args.clobber,
            ..ConvertOptions::default()
        };
        let emitted =
            convert_frame(&frame, &weights, &destination, &options).map_err(|e| e.to_string())?;
        println!(
            "CONVERTED\t{}\t{:.3}s\t{} bytes\t{} written\t{} absent",
            emitted.path.display(),
            emitted.seconds,
            emitted.bytes,
            emitted.written.len(),
            emitted.absent.len()
        );
        records.push(format!(
            "{{\"output\": \"{}\", \"output_sha256\": \"{}\", \"output_bytes\": {}, \"valid_time\": \"{}\", \"simulation_start\": \"{}\", \"lead_seconds\": {}, \"convert_seconds\": {}, \"history_sha256\": \"{}\", \"init_sha256\": \"{}\", \"max_latitude_error_degrees\": {:e}, \"max_longitude_error_degrees\": {:e}, \"written_variables\": [{}], \"absent_wrf_fields\": [{}]}}",
            json_escape(&emitted.path.display().to_string()),
            frame.history_sha256,
            emitted.bytes,
            emitted.valid_time.iso(),
            emitted.simulation_start.iso(),
            emitted.lead_seconds,
            emitted.seconds,
            frame.history_sha256,
            frame.init_sha256.clone().unwrap_or_default(),
            dlat,
            dlon,
            emitted
                .written
                .iter()
                .map(|n| format!("\"{n}\""))
                .collect::<Vec<_>>()
                .join(", "),
            emitted
                .absent
                .iter()
                .map(|n| format!("\"{n}\""))
                .collect::<Vec<_>>()
                .join(", "),
        ));
    }

    if let Some(path) = &args.json {
        let body = format!(
            "{{\n  \"schema\": \"{}\",\n  \"engine\": \"rw-mpas {} (rust)\",\n  \"window\": \"{}\",\n  \"window_spec\": {},\n  \"mesh_sha256\": \"{}\",\n  \"weights_sha256\": \"{}\",\n  \"resample_method\": \"{}\",\n  \"field_set\": \"{}\",\n  \"weights_build_seconds\": {},\n  \"frames\": [\n    {}\n  ]\n}}\n",
            rw_mpas::convert::BRIDGE_SCHEMA,
            env!("CARGO_PKG_VERSION"),
            args.window,
            window.spec_json(),
            weights.mesh_sha256,
            weights.weights_sha256,
            rw_mpas::weights::RESAMPLE_METHOD,
            args.field_set,
            weights.build_seconds,
            records.join(",\n    ")
        );
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, body).map_err(|e| e.to_string())?;
        println!("REPORT\t{}", path.display());
    }
    println!(
        "FINISHED\tframes={}\telapsed_s={:.3}",
        args.history.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    if argv.iter().any(|a| a == "--abi") {
        println!("{ABI}");
        return ExitCode::SUCCESS;
    }
    if argv.is_empty() {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    }
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("rw_mpas_convert: {message}");
            ExitCode::FAILURE
        }
    }
}
