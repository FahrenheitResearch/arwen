//! The CLI frame selector applies to windowed products as well as static
//! products, while unselected earlier frames remain accumulation context.
mod stored_plane_fixture;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("rw-windowed-frames-{}-{nonce}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn render(root: &Path, tag: &str, frames: &str, inputs: &[PathBuf]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rw_wrfbatch"))
        .args([
            "--products",
            "qpf_1h",
            "--frames",
            frames,
            "--width",
            "480",
            "--height",
            "360",
        ])
        .arg("--store-root")
        .arg(root.join(format!("store-{tag}")))
        .arg("--out-dir")
        .arg(root.join(format!("out-{tag}")))
        .args(inputs)
        .env("GPUWM_NO_LOCAL_GPU", "1")
        .env("CUDA_VISIBLE_DEVICES", "-1")
        .env("RUSTWX_BATCH_RENDER_THREADS", "1")
        .output()
        .unwrap()
}

fn rendered(stdout: &str) -> Vec<PathBuf> {
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("RENDERED qpf_1h ").map(PathBuf::from))
        .collect()
}

#[test]
fn all_frames_and_selected_frame_use_their_actual_hour_with_baseline_context() {
    let scratch = Scratch::new();
    let inputs: Vec<PathBuf> = (0..=6)
        .map(|hour| {
            stored_plane_fixture::write_rain_frame(
                &scratch.0,
                hour * 3600,
                (hour * (hour + 1)) as f32,
            )
        })
        .collect();
    let all = render(&scratch.0, "all", "all", &inputs);
    let stdout = String::from_utf8_lossy(&all.stdout);
    assert!(
        all.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&all.stderr)
    );
    let paths = rendered(&stdout);
    assert_eq!(paths.len(), 6, "{stdout}");
    for (hour, path) in (1..=6).zip(&paths) {
        assert!(
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .contains(&format!("_f{hour:03}_")),
            "{}",
            path.display()
        );
        assert!(path.is_file());
    }
    // The skip names the file of ITS frame (F000), not the last input.
    assert!(
        stdout.contains(&format!("SKIPPED qpf_1h {}: F000:", inputs[0].display())),
        "{stdout}"
    );

    let selected = render(&scratch.0, "selected", "2", &inputs);
    let selected_stdout = String::from_utf8_lossy(&selected.stdout);
    assert!(
        selected.status.success(),
        "{selected_stdout}\n{}",
        String::from_utf8_lossy(&selected.stderr)
    );
    let selected_paths = rendered(&selected_stdout);
    assert_eq!(selected_paths.len(), 1, "{selected_stdout}");
    assert!(
        selected_paths[0]
            .file_name()
            .unwrap()
            .to_string_lossy()
            .contains("_f002_")
    );
    assert_eq!(
        std::fs::read(&paths[1]).unwrap(),
        std::fs::read(&selected_paths[0]).unwrap()
    );
}

#[test]
fn subhourly_ordinal_slots_are_not_treated_as_forecast_hours() {
    // Frames every 15 minutes put the store on the exact-time axis: slot 4
    // is the frame at +1 h.  Its 1 h window is drawn as F001, never as
    // F004, and the frames between hours end no window and add no skip.
    let scratch = Scratch::new();
    let inputs: Vec<PathBuf> = (0..=4)
        .map(|slot| stored_plane_fixture::write_rain_frame(&scratch.0, slot * 900, slot as f32))
        .collect();
    let output = render(&scratch.0, "exact", "all", &inputs);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    let paths = rendered(&stdout);
    assert_eq!(paths.len(), 1, "{stdout}");
    let name = paths[0].file_name().unwrap().to_string_lossy().to_string();
    assert!(
        name.contains("_f001_") && !name.contains("_f004_"),
        "{name}"
    );
    assert!(paths[0].is_file());
    let skips: Vec<&str> = stdout
        .lines()
        .filter(|line| line.starts_with("SKIPPED qpf_1h "))
        .collect();
    assert_eq!(skips.len(), 1, "only the analysis frame skips: {stdout}");
    assert!(
        skips[0].contains(&format!("{}: F000:", inputs[0].display())),
        "{stdout}"
    );
}
