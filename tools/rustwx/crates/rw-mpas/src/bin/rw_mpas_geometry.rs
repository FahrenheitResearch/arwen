//! Derived vector geometry and exact RBF reconstruction for the hex init door.

use std::process::ExitCode;

pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn main() -> ExitCode {
    let _stamp = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args == ["--abi"] {
        println!("{}",rw_mpas::geometry::ABI_MARKER);
        return ExitCode::SUCCESS;
    }
    if args == ["--help"] {
        println!("{}\nstdin: HEXGEO1 little-endian arrays; stdout: raw derived arrays",rw_mpas::geometry::ABI_MARKER);
        return ExitCode::SUCCESS;
    }
    if args != ["--protocol","hex-geometry-v1"] {
        eprintln!("usage: {}",rw_mpas::geometry::ABI_MARKER);
        return ExitCode::FAILURE;
    }
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    match rw_mpas::geometry::run(stdin.lock(),std::io::BufWriter::new(stdout.lock())) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => { eprintln!("geometry: {error}"); ExitCode::FAILURE }
    }
}
