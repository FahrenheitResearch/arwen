//! Small compatibility executable. The shared renderer receives the original
//! argv, streams and working directory, including the invoked tool name.
mod cli_contracts;
#[cfg(windows)]
mod launcher_windows;

fn main() {
    let marker = match env!("CARGO_BIN_NAME") {
        "rw_ensbatch" => cli_contracts::ENSBATCH,
        "rw_compare" => cli_contracts::COMPARE,
        "rw_verify" => cli_contracts::VERIFY,
        "rw_scorecard" => cli_contracts::SCORECARD,
        "rw_nowcast_frames" => cli_contracts::NOWCAST_FRAMES,
        _ => unreachable!(),
    };
    std::hint::black_box(marker);
    // These two tools also carry a dependency's format marker. Doctor and
    // bundle staging read that marker rather than the command's --abi line.
    match env!("CARGO_BIN_NAME") {
        "rw_ensbatch" => { std::hint::black_box(cli_contracts::ENSBATCH_STATIC); }
        "rw_compare" => { std::hint::black_box(cli_contracts::COMPARE_STATIC); }
        _ => {}
    }
    std::hint::black_box(concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV")));
    let result = launch();
    if let Err(error) = result {
        eprintln!("{}: cannot launch rw_wrfbatch: {error}", env!("CARGO_BIN_NAME"));
        std::process::exit(1);
    }
}

fn launch() -> std::io::Result<()> {
    let executable = std::env::current_exe()?.with_file_name(if cfg!(windows) {
        "rw_wrfbatch.exe"
    } else {
        "rw_wrfbatch"
    });
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let mut args = std::env::args_os();
        let invoked = args.next().unwrap_or_else(|| env!("CARGO_BIN_NAME").into());
        // exec retains PID, inherited descriptors and signal exit status.
        Err(std::process::Command::new(executable).arg0(invoked).args(args).exec())
    }
    #[cfg(windows)]
    {
        launcher_windows::launch(&executable)
    }
}
